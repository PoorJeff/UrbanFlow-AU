from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from urbanflow.api.postgres import PostgresSensorHistoryRepository
from urbanflow.api.services import DataStoreUnavailableError, HistoryRecord
from urbanflow.database.engine import create_database_engine
from urbanflow.database.models import Base
from urbanflow.database.repositories import upsert_hourly_rows, upsert_sensor_rows

SMOKE_DATABASE_URL_ENV_VAR = "URBANFLOW_SMOKE_DATABASE_URL"

_SAFE_SCHEMA_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


@dataclass(frozen=True)
class PostgresApiRepositorySmokeResult:
    schema_name: str
    all_sensor_location_ids: list[int]
    active_sensor_location_ids: list[int]
    history_count: int


def validate_smoke_schema_name(schema_name: str) -> str:
    if not _SAFE_SCHEMA_NAME_PATTERN.fullmatch(schema_name):
        raise ValueError(
            "Smoke schema name must be a safe PostgreSQL identifier: "
            "lowercase letters, digits, and underscores only, starting with a letter."
        )
    return schema_name


def run_postgres_api_repository_smoke(
    database_url: str,
    *,
    schema_name: str | None = None,
) -> PostgresApiRepositorySmokeResult:
    schema = validate_smoke_schema_name(schema_name or _temporary_schema_name())
    quoted_schema = _quote_identifier(schema)
    engine = create_database_engine(database_url)
    schema_created = False

    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f"CREATE SCHEMA {quoted_schema}")
            schema_created = True
            connection.exec_driver_sql(f"SET search_path TO {quoted_schema}")
            Base.metadata.create_all(connection)
            session_factory = sessionmaker(
                bind=connection,
                autoflush=False,
                expire_on_commit=False,
            )
            source_observed_at = datetime(
                2025,
                1,
                1,
                7,
                tzinfo=ZoneInfo("Australia/Melbourne"),
            )
            with session_factory() as session:
                upsert_sensor_rows(
                    session,
                    [
                        _sensor_smoke_row(location_id=999001, status="A"),
                        _sensor_smoke_row(location_id=999002, status="I"),
                    ],
                )
                upsert_hourly_rows(session, [_hourly_smoke_row(source_observed_at)])

            repository = PostgresSensorHistoryRepository(session_factory)
            all_sensor_location_ids = [
                sensor.location_id for sensor in repository.list_sensors(active_only=False)
            ]
            active_sensor_location_ids = [
                sensor.location_id for sensor in repository.list_sensors(active_only=True)
            ]
            history = repository.get_history(
                999001,
                source_observed_at - timedelta(hours=1),
                source_observed_at + timedelta(hours=1),
            )
            _assert_expected_repository_data(
                all_sensor_location_ids=all_sensor_location_ids,
                active_sensor_location_ids=active_sensor_location_ids,
                history=history,
                source_observed_at=source_observed_at,
            )
        return PostgresApiRepositorySmokeResult(
            schema_name=schema,
            all_sensor_location_ids=all_sensor_location_ids,
            active_sensor_location_ids=active_sensor_location_ids,
            history_count=len(history),
        )
    finally:
        try:
            if schema_created:
                with engine.begin() as connection:
                    connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {quoted_schema} CASCADE")
        finally:
            engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test UrbanFlow AU PostgreSQL API repositories against a local database."
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--schema-name",
        default=None,
        help="Optional temporary schema name for debugging. Defaults to a generated name.",
    )
    return parser


def main(argv: list[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        database_url = _database_url(args.database_url, environ=environ)
        result = run_postgres_api_repository_smoke(
            database_url,
            schema_name=args.schema_name,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (DataStoreUnavailableError, SQLAlchemyError) as exc:
        print(f"PostgreSQL API repository smoke test failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(result), sort_keys=True))
    return 0


def _temporary_schema_name() -> str:
    return f"urbanflow_api_smoke_{uuid4().hex[:12]}"


def _quote_identifier(identifier: str) -> str:
    return f'"{validate_smoke_schema_name(identifier)}"'


def _sensor_smoke_row(*, location_id: int, status: str) -> dict[str, object]:
    return {
        "location_id": location_id,
        "sensor_name": f"API Smoke Test Sensor {location_id}",
        "sensor_description": "Synthetic PostgreSQL API repository smoke-test sensor",
        "latitude": -37.8136,
        "longitude": 144.9631,
        "installation_date": date(2025, 1, 1),
        "status": status,
    }


def _hourly_smoke_row(observed_at: datetime) -> dict[str, object]:
    return {
        "location_id": 999001,
        "observed_at": observed_at,
        "source_sensing_date": observed_at.date(),
        "source_hourday": observed_at.hour,
        "pedestrian_count": 42,
        "direction_1_count": 20,
        "direction_2_count": 22,
        "source_snapshot_path": "smoke://postgres-api-repository",
    }


def _assert_expected_repository_data(
    *,
    all_sensor_location_ids: list[int],
    active_sensor_location_ids: list[int],
    history: list[HistoryRecord],
    source_observed_at: datetime,
) -> None:
    if all_sensor_location_ids != [999001, 999002]:
        raise AssertionError(f"Unexpected all-sensor IDs: {all_sensor_location_ids}")
    if active_sensor_location_ids != [999001]:
        raise AssertionError(f"Unexpected active-sensor IDs: {active_sensor_location_ids}")
    if len(history) != 1:
        raise AssertionError(f"Unexpected history row count: {len(history)}")
    record = history[0]
    if record.pedestrian_count != 42:
        raise AssertionError(f"Unexpected pedestrian count: {record.pedestrian_count}")
    if record.observed_at.tzinfo is None or record.observed_at.utcoffset() is None:
        raise AssertionError("Repository history timestamp must be timezone-aware")
    if record.observed_at.astimezone(UTC) != source_observed_at.astimezone(UTC):
        raise AssertionError("Repository history timestamp did not preserve the source instant")


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
            "PostgreSQL API repository smoke database URL is required. "
            f"Pass --database-url or set {SMOKE_DATABASE_URL_ENV_VAR}."
        )
    return database_url.strip()
