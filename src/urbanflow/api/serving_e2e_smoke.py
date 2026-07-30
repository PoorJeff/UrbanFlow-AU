from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import uuid4

import httpx
import pandas as pd
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.orm import sessionmaker

from urbanflow.api import postgres_smoke
from urbanflow.api.services import API_MAX_DATA_AGE_HOURS_ENV_VAR
from urbanflow.dashboard.client import DashboardApiClient
from urbanflow.database.config import DATABASE_URL_ENV_VAR
from urbanflow.database.engine import create_database_engine
from urbanflow.database.models import Base
from urbanflow.database.repositories import upsert_hourly_rows, upsert_sensor_rows
from urbanflow.database.time import MELBOURNE_TZ
from urbanflow.features.supervised import build_supervised_frame
from urbanflow.modeling.lightgbm import LightGBMModelConfig
from urbanflow.modeling.lightgbm_artifact import (
    HolidayCalendar,
    export_lightgbm_artifact,
)
from urbanflow.modeling.supervised_csv import read_supervised_csv, sha256_file

SMOKE_DATABASE_URL_ENV_VAR = "URBANFLOW_SMOKE_DATABASE_URL"

_MODEL_ARTIFACT_PATH_ENV_VAR = "URBANFLOW_API_MODEL_ARTIFACT_PATH"
_SMOKE_LOCATION_ID = 999001
_HISTORY_LENGTH = 192
_FORECAST_HORIZON = 24
_MAX_DATA_AGE_HOURS = 2
_STARTUP_TIMEOUT_SECONDS = 30.0
_STARTUP_POLL_INTERVAL_SECONDS = 0.1
_HTTP_POLL_TIMEOUT_SECONDS = 1.0
_PROCESS_STOP_TIMEOUT_SECONDS = 10.0
_MAX_LOG_TAIL_CHARS = 2_000


class ServingE2ESmokeError(RuntimeError):
    """Raised when configured API serving does not satisfy the smoke contract."""


class _Process(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float) -> int: ...

    def kill(self) -> None: ...


@dataclass(frozen=True)
class ServingE2ESmokeResult:
    schema_name: str
    location_id: int
    health_status: str
    data_cutoff_at: str
    model_version: str
    history_count: int
    forecast_horizons: list[int]


@dataclass(frozen=True)
class _SmokeFixture:
    artifact_path: Path
    history_start: datetime
    history_end: datetime
    expected_cutoff: datetime
    expected_model_version: str


def run_serving_e2e_smoke(
    database_url: str,
    *,
    schema_name: str | None = None,
) -> ServingE2ESmokeResult:
    return _run_serving_e2e_smoke(database_url, schema_name=schema_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test UrbanFlow AU configured API serving end to end."
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--schema-name",
        default=None,
        help="Optional temporary schema name for debugging. Defaults to a generated name.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        database_url = _database_url(args.database_url, environ=environ)
        result = run_serving_e2e_smoke(database_url, schema_name=args.schema_name)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        print("Configured serving smoke failed.", file=sys.stderr)
        return 1

    print(json.dumps(asdict(result), sort_keys=True))
    return 0


def _run_serving_e2e_smoke(
    database_url: str,
    *,
    schema_name: str | None = None,
    engine_factory: Callable[[str], Engine] | None = None,
    create_tables: Callable[[Connection], None] | None = None,
    fixture_builder: Callable[[Connection, Path, datetime], _SmokeFixture] | None = None,
    process_factory: Callable[..., _Process] | None = None,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    free_port_allocator: Callable[[], int] | None = None,
    http_poller: Callable[[str, float], bool] | None = None,
    client_factory: Callable[[str], DashboardApiClient] | None = None,
    response_assertion: (
        Callable[[DashboardApiClient, _SmokeFixture, str], ServingE2ESmokeResult] | None
    ) = None,
    temporary_log_factory: Callable[[], BinaryIO] | None = None,
) -> ServingE2ESmokeResult:
    schema = postgres_smoke.validate_smoke_schema_name(schema_name or _temporary_schema_name())
    quoted_schema = _quote_identifier(schema)
    resolved_engine_factory = engine_factory or create_database_engine
    resolved_create_tables = create_tables or Base.metadata.create_all
    resolved_fixture_builder = fixture_builder or _build_smoke_fixture
    resolved_process_factory = process_factory or subprocess.Popen
    resolved_monotonic = monotonic or time.monotonic
    resolved_sleep = sleep or time.sleep
    resolved_free_port_allocator = free_port_allocator or _allocate_free_port
    resolved_http_poller = http_poller or _poll_health
    resolved_client_factory = client_factory or DashboardApiClient
    resolved_response_assertion = response_assertion or _assert_serving_responses
    resolved_temporary_log_factory = temporary_log_factory or _temporary_log_file

    engine: Engine | None = None
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    log_file: BinaryIO | None = None
    process: _Process | None = None
    client: DashboardApiClient | None = None
    schema_created = False
    cleanup_error: Exception | None = None

    try:
        engine = resolved_engine_factory(database_url)
        temporary_directory = tempfile.TemporaryDirectory(prefix="urbanflow-serving-e2e-smoke-")
        temporary_path = Path(temporary_directory.name)
        with engine.begin() as connection:
            connection.exec_driver_sql(f"CREATE SCHEMA {quoted_schema}")
            schema_created = True
            connection.exec_driver_sql(f"SET search_path TO {quoted_schema}")
            resolved_create_tables(connection)
            fixture = resolved_fixture_builder(
                connection,
                temporary_path,
                datetime.now(UTC),
            )

        child_database_url = _schema_database_url(database_url, schema)
        child_environment = _child_environment(
            child_database_url,
            fixture.artifact_path,
        )
        port = resolved_free_port_allocator()
        base_url = f"http://127.0.0.1:{port}"
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "urbanflow.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        log_file = resolved_temporary_log_factory()
        process = resolved_process_factory(
            command,
            env=child_environment,
            stdout=log_file,
            stderr=log_file,
            shell=False,
        )
        _wait_for_api_ready(
            process,
            f"{base_url}/health",
            log_file,
            monotonic=resolved_monotonic,
            sleep=resolved_sleep,
            http_poller=resolved_http_poller,
            sensitive_values=(database_url, child_database_url, str(fixture.artifact_path)),
        )
        client = resolved_client_factory(base_url)
        result = resolved_response_assertion(client, fixture, schema)
    except ServingE2ESmokeError:
        raise
    except Exception as exc:
        raise ServingE2ESmokeError("Configured serving smoke failed.") from exc
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if process is not None:
            try:
                _stop_process(process)
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if log_file is not None:
            try:
                log_file.close()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if engine is not None and schema_created:
            try:
                with engine.begin() as connection:
                    connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {quoted_schema} CASCADE")
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if engine is not None:
            try:
                engine.dispose()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if temporary_directory is not None:
            try:
                temporary_directory.cleanup()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None and sys.exc_info()[0] is None:
            raise ServingE2ESmokeError(
                "Configured serving smoke cleanup failed."
            ) from cleanup_error

    return result


def _temporary_schema_name() -> str:
    return f"urbanflow_serving_e2e_{uuid4().hex[:12]}"


def _quote_identifier(identifier: str) -> str:
    return f'"{postgres_smoke.validate_smoke_schema_name(identifier)}"'


def _schema_database_url(database_url: str, schema: str) -> str:
    validated_schema = postgres_smoke.validate_smoke_schema_name(schema)
    child_url = make_url(database_url).update_query_dict(
        {"options": f"-csearch_path={validated_schema}"}
    )
    return child_url.render_as_string(hide_password=False)


def _child_environment(child_database_url: str, artifact_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("URBANFLOW_"):
            del environment[key]
    environment.update(
        {
            DATABASE_URL_ENV_VAR: child_database_url,
            _MODEL_ARTIFACT_PATH_ENV_VAR: str(artifact_path),
            API_MAX_DATA_AGE_HOURS_ENV_VAR: str(_MAX_DATA_AGE_HOURS),
        }
    )
    return environment


def _wait_for_api_ready(
    process: _Process,
    health_url: str,
    log_file: BinaryIO,
    *,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    http_poller: Callable[[str, float], bool],
    sensitive_values: tuple[str, ...],
) -> None:
    deadline = monotonic() + _STARTUP_TIMEOUT_SECONDS
    maximum_polls = int(_STARTUP_TIMEOUT_SECONDS / _STARTUP_POLL_INTERVAL_SECONDS) + 2
    for _ in range(maximum_polls):
        if process.poll() is not None:
            tail = _safe_log_tail(log_file, sensitive_values=sensitive_values)
            raise ServingE2ESmokeError(
                f"Configured serving process exited before readiness. Log tail: {tail}"
            )
        if http_poller(health_url, _HTTP_POLL_TIMEOUT_SECONDS):
            return
        remaining = deadline - monotonic()
        if remaining <= 0:
            tail = _safe_log_tail(log_file, sensitive_values=sensitive_values)
            raise ServingE2ESmokeError(
                f"Configured serving process timed out before readiness. Log tail: {tail}"
            )
        sleep(min(_STARTUP_POLL_INTERVAL_SECONDS, remaining))

    tail = _safe_log_tail(log_file, sensitive_values=sensitive_values)
    raise ServingE2ESmokeError(
        f"Configured serving process timed out before readiness. Log tail: {tail}"
    )


def _stop_process(process: _Process) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)


def _safe_log_tail(
    log_file: BinaryIO,
    *,
    sensitive_values: tuple[str, ...],
) -> str:
    log_file.flush()
    log_file.seek(0, os.SEEK_END)
    size = log_file.tell()
    log_file.seek(max(0, size - _MAX_LOG_TAIL_CHARS))
    tail = log_file.read(_MAX_LOG_TAIL_CHARS).decode("utf-8", errors="replace")
    for value in sensitive_values:
        if value:
            tail = tail.replace(value, "[redacted]")
    tail = re.sub(
        r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+",
        "[redacted-url]",
        tail,
    )
    tail = re.sub(
        r"(?i)\b(?:password|passwd|secret|token)\s*[:=]\s*[^\s]+",
        "[redacted-credential]",
        tail,
    )
    tail = re.sub(
        r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"']+",
        "[redacted-path]",
        tail,
    )
    return tail[-_MAX_LOG_TAIL_CHARS:].strip() or "<empty>"


def _temporary_log_file() -> BinaryIO:
    return tempfile.TemporaryFile(mode="w+b")


def _allocate_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _poll_health(health_url: str, timeout_seconds: float) -> bool:
    try:
        response = httpx.get(health_url, timeout=timeout_seconds)
    except httpx.RequestError:
        return False
    return response.status_code == 200


def _build_smoke_fixture(
    connection: Connection,
    temporary_directory: Path,
    now: datetime,
) -> _SmokeFixture:
    source_timestamps, observations = _smoke_observations(now)
    calendar = _smoke_holiday_calendar(source_timestamps)
    calendar_path = temporary_directory / "holiday_calendar.json"
    _write_smoke_holiday_calendar(calendar_path, calendar)
    calendar = HolidayCalendar.from_json_file(calendar_path)
    supervised_csv_path = temporary_directory / "supervised.csv"
    build_supervised_frame(
        observations,
        public_holidays=calendar.public_holidays,
    ).to_csv(supervised_csv_path, index=False)
    artifact_path = temporary_directory / "artifact"
    manifest = export_lightgbm_artifact(
        read_supervised_csv(supervised_csv_path),
        source_csv_sha256=sha256_file(supervised_csv_path),
        output_directory=artifact_path,
        holiday_calendar=calendar,
        model_config=LightGBMModelConfig(
            n_estimators=5,
            min_child_samples=1,
        ),
    )

    session_factory = sessionmaker(
        bind=connection,
        autoflush=False,
        expire_on_commit=False,
    )
    with session_factory() as session:
        upsert_sensor_rows(session, [_sensor_smoke_row()])
        upsert_hourly_rows(session, _hourly_smoke_rows(source_timestamps))
        session.commit()

    expected_cutoff = source_timestamps[-1].to_pydatetime()
    return _SmokeFixture(
        artifact_path=artifact_path,
        history_start=source_timestamps[0].to_pydatetime(),
        history_end=expected_cutoff + timedelta(hours=1),
        expected_cutoff=expected_cutoff,
        expected_model_version=manifest.model_version,
    )


def _smoke_observations(now: datetime) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Smoke clock must be timezone-aware.")
    final_timestamp = pd.Timestamp(now.astimezone(MELBOURNE_TZ)).floor("h")
    timestamps = pd.date_range(
        end=final_timestamp,
        periods=_HISTORY_LENGTH,
        freq="h",
    )
    counts = [100 + (index % 24) * 3 + index // 24 for index in range(_HISTORY_LENGTH)]
    observations = pd.DataFrame(
        {
            "location_id": [_SMOKE_LOCATION_ID] * _HISTORY_LENGTH,
            "observed_at": timestamps,
            "pedestrian_count": counts,
        }
    )
    return timestamps, observations


def _smoke_holiday_calendar(
    timestamps: pd.DatetimeIndex,
) -> HolidayCalendar:
    final_target = (
        timestamps[-1].to_pydatetime().astimezone(UTC) + timedelta(hours=_FORECAST_HORIZON)
    ).astimezone(MELBOURNE_TZ)
    return HolidayCalendar(
        coverage_start=timestamps[0].date(),
        coverage_end=final_target.date(),
        public_holidays=(timestamps[0].date(),),
    )


def _write_smoke_holiday_calendar(
    path: Path,
    calendar: HolidayCalendar,
) -> None:
    path.write_text(
        json.dumps(
            {
                "coverage_start": calendar.coverage_start.isoformat(),
                "coverage_end": calendar.coverage_end.isoformat(),
                "public_holidays": [
                    public_holiday.isoformat() for public_holiday in calendar.public_holidays
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sensor_smoke_row() -> dict[str, object]:
    return {
        "location_id": _SMOKE_LOCATION_ID,
        "sensor_name": "Configured Serving Smoke Test Sensor",
        "sensor_description": "Synthetic configured-serving smoke-test sensor",
        "latitude": -37.8136,
        "longitude": 144.9631,
        "installation_date": date.today(),
        "status": "A",
    }


def _hourly_smoke_rows(
    timestamps: pd.DatetimeIndex,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, observed_at in enumerate(timestamps):
        pedestrian_count = 100 + (index % 24) * 3 + index // 24
        rows.append(
            {
                "location_id": _SMOKE_LOCATION_ID,
                "observed_at": observed_at.to_pydatetime(),
                "source_sensing_date": observed_at.date(),
                "source_hourday": observed_at.hour,
                "pedestrian_count": pedestrian_count,
                "direction_1_count": pedestrian_count // 2,
                "direction_2_count": pedestrian_count - pedestrian_count // 2,
                "source_snapshot_path": "smoke://configured-serving",
            }
        )
    return rows


def _assert_serving_responses(
    client: DashboardApiClient,
    fixture: _SmokeFixture,
    schema: str,
) -> ServingE2ESmokeResult:
    health = client.get_health()
    component_statuses = (
        health.components.api_process.status,
        health.components.model_provider.status,
        health.components.data_store.status,
        health.components.data_freshness.status,
    )
    if health.status != "ok" or component_statuses != ("available",) * 4:
        raise ServingE2ESmokeError("Configured serving health is not ready.")
    if (
        health.data_cutoff_at is None
        or health.data_cutoff_at.astimezone(UTC) != fixture.expected_cutoff.astimezone(UTC)
        or health.model_version != fixture.expected_model_version
    ):
        raise ServingE2ESmokeError("Configured serving health metadata does not match the fixture.")

    sensors = client.list_sensors(active_only=True)
    if (
        sensors.meta.active_only is not True
        or sensors.meta.count != 1
        or len(sensors.data) != 1
        or sensors.data[0].location_id != _SMOKE_LOCATION_ID
    ):
        raise ServingE2ESmokeError(
            "Configured serving did not return exactly the seeded active sensor."
        )

    history = client.get_history(
        _SMOKE_LOCATION_ID,
        start=fixture.history_start,
        end=fixture.history_end,
    )
    history_timestamps = [point.observed_at for point in history.data]
    normalized_history_timestamps = [timestamp.astimezone(UTC) for timestamp in history_timestamps]
    expected_history_timestamps = [
        fixture.history_start.astimezone(UTC) + timedelta(hours=index)
        for index in range(_HISTORY_LENGTH)
    ]
    if (
        len(history.data) != _HISTORY_LENGTH
        or normalized_history_timestamps != expected_history_timestamps
    ):
        raise ServingE2ESmokeError(
            "Configured serving history does not match the seeded observations."
        )

    forecast = client.get_forecast(
        _SMOKE_LOCATION_ID,
        horizon=_FORECAST_HORIZON,
    )
    forecast_horizons = [prediction.forecast_horizon for prediction in forecast.predictions]
    if (
        forecast_horizons != list(range(1, _FORECAST_HORIZON + 1))
        or forecast.data_cutoff_at.astimezone(UTC) != fixture.expected_cutoff.astimezone(UTC)
        or forecast.model_version != fixture.expected_model_version
    ):
        raise ServingE2ESmokeError("Configured serving forecast does not match the fixture.")

    return ServingE2ESmokeResult(
        schema_name=schema,
        location_id=_SMOKE_LOCATION_ID,
        health_status=health.status,
        data_cutoff_at=forecast.data_cutoff_at.isoformat(),
        model_version=fixture.expected_model_version,
        history_count=len(history.data),
        forecast_horizons=forecast_horizons,
    )


def _database_url(
    explicit_database_url: str | None,
    *,
    environ: Mapping[str, str] | None,
) -> str:
    values = os.environ if environ is None else environ
    database_url = (
        explicit_database_url
        if explicit_database_url is not None
        else values.get(SMOKE_DATABASE_URL_ENV_VAR)
    )
    if database_url is None or not database_url.strip():
        raise ValueError(
            "Configured serving smoke database URL is required. "
            f"Pass --database-url or set {SMOKE_DATABASE_URL_ENV_VAR}."
        )
    return database_url.strip()
