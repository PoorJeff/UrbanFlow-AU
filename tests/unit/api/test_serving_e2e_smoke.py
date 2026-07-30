from __future__ import annotations

import io
import json
import subprocess
import traceback
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.engine import make_url

from urbanflow.api import postgres_smoke, serving_e2e_smoke


def test_generated_schema_is_safe_and_dangerous_schema_is_rejected_before_engine_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_schema = serving_e2e_smoke._temporary_schema_name()
    engine_urls: list[str] = []
    monkeypatch.setattr(
        serving_e2e_smoke,
        "create_database_engine",
        lambda database_url: engine_urls.append(database_url),
    )

    assert postgres_smoke.validate_smoke_schema_name(generated_schema) == generated_schema
    with pytest.raises(ValueError, match="safe PostgreSQL identifier"):
        serving_e2e_smoke.run_serving_e2e_smoke(
            "postgresql+psycopg://user:secret@localhost/urbanflow",
            schema_name="urbanflow;drop schema public",
        )
    assert engine_urls == []


def test_cli_returns_two_and_names_smoke_environment_when_database_url_is_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = serving_e2e_smoke.main([], environ={})

    assert exit_code == 2
    captured = capsys.readouterr()
    assert serving_e2e_smoke.SMOKE_DATABASE_URL_ENV_VAR in captured.err
    assert captured.out == ""


def test_schema_database_url_preserves_query_and_encodes_validated_search_path() -> None:
    base_url = (
        "postgresql+psycopg://urbanflow:p%40ss@localhost:5432/urbanflow"
        "?sslmode=require&application_name=urbanflow-smoke"
    )

    child_url = serving_e2e_smoke._schema_database_url(
        base_url,
        "urbanflow_serving_e2e_test",
    )

    parsed = make_url(child_url)
    assert parsed.username == "urbanflow"
    assert parsed.password == "p@ss"
    assert parsed.host == "localhost"
    assert parsed.port == 5432
    assert parsed.database == "urbanflow"
    assert parsed.query == {
        "sslmode": "require",
        "application_name": "urbanflow-smoke",
        "options": "-csearch_path=urbanflow_serving_e2e_test",
    }
    assert "options=-csearch_path%3Durbanflow_serving_e2e_test" in child_url


def test_schema_database_url_preserves_existing_non_search_path_options() -> None:
    base_url = (
        "postgresql+psycopg://urbanflow:secret@localhost:5432/urbanflow"
        "?sslmode=require"
        "&options=-cstatement_timeout%3D5000%20-csearch_path%3Dpublic"
    )

    child_url = serving_e2e_smoke._schema_database_url(
        base_url,
        "urbanflow_serving_e2e_test",
    )

    parsed = make_url(child_url)
    assert parsed.query["sslmode"] == "require"
    assert parsed.query["options"] == (
        "-cstatement_timeout=5000 -csearch_path=urbanflow_serving_e2e_test"
    )


def test_child_environment_scrubs_inherited_urbanflow_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("UNRELATED_SETTING", "kept")
    monkeypatch.setenv("URBANFLOW_DATABASE_URL", "inherited-database")
    monkeypatch.setenv("URBANFLOW_API_MODEL_ARTIFACT_PATH", "inherited-artifact")
    monkeypatch.setenv("URBANFLOW_API_MAX_DATA_AGE_HOURS", "999")
    monkeypatch.setenv("URBANFLOW_OTHER_SECRET", "must-be-removed")
    monkeypatch.setenv("UVICORN_RELOAD", "true")
    monkeypatch.setenv("UVICORN_WORKERS", "4")
    monkeypatch.setenv("UVICORN_PORT", "9999")
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    artifact_path = tmp_path / "artifact"

    child_environment = serving_e2e_smoke._child_environment(
        "postgresql+psycopg://localhost/urbanflow?options=safe",
        artifact_path,
    )

    assert child_environment["UNRELATED_SETTING"] == "kept"
    assert not any(key.startswith("UVICORN_") for key in child_environment)
    assert "WEB_CONCURRENCY" not in child_environment
    assert {
        key: value for key, value in child_environment.items() if key.startswith("URBANFLOW_")
    } == {
        "URBANFLOW_DATABASE_URL": ("postgresql+psycopg://localhost/urbanflow?options=safe"),
        "URBANFLOW_API_MODEL_ARTIFACT_PATH": str(artifact_path),
        "URBANFLOW_API_MAX_DATA_AGE_HOURS": "2",
    }
    assert int(child_environment["URBANFLOW_API_MAX_DATA_AGE_HOURS"]) > 0


def test_result_serialization_contains_no_url_secret_or_artifact_path() -> None:
    result = serving_e2e_smoke.ServingE2ESmokeResult(
        schema_name="urbanflow_serving_e2e_test",
        location_id=999001,
        health_status="ok",
        data_cutoff_at="2026-07-30T02:00:00+00:00",
        model_version="lightgbm-0123456789ab",
        history_count=192,
        forecast_horizons=list(range(1, 25)),
    )

    payload = asdict(result)
    encoded = json.dumps(payload, sort_keys=True)

    assert set(payload) == {
        "schema_name",
        "location_id",
        "health_status",
        "data_cutoff_at",
        "model_version",
        "history_count",
        "forecast_horizons",
    }
    assert "postgresql" not in encoded
    assert "secret" not in encoded
    assert "artifact" not in encoded


def test_response_assertion_rejects_ordered_history_that_does_not_match_seeded_hours(
    tmp_path: Path,
) -> None:
    cutoff = datetime(2026, 7, 30, 2, tzinfo=UTC)
    history_start = cutoff - timedelta(hours=191)
    history_timestamps = [history_start + timedelta(hours=index) for index in range(192)]
    history_timestamps[100] += timedelta(minutes=30)
    available = SimpleNamespace(status="available")
    client = SimpleNamespace(
        get_health=lambda: SimpleNamespace(
            status="ok",
            components=SimpleNamespace(
                api_process=available,
                model_provider=available,
                data_store=available,
                data_freshness=available,
            ),
            data_cutoff_at=cutoff,
            model_version="lightgbm-0123456789ab",
        ),
        list_sensors=lambda **_kwargs: SimpleNamespace(
            meta=SimpleNamespace(active_only=True, count=1),
            data=[SimpleNamespace(location_id=999001)],
        ),
        get_history=lambda *_args, **_kwargs: SimpleNamespace(
            data=[SimpleNamespace(observed_at=timestamp) for timestamp in history_timestamps]
        ),
        get_forecast=lambda *_args, **_kwargs: SimpleNamespace(
            predictions=[SimpleNamespace(forecast_horizon=horizon) for horizon in range(1, 25)],
            data_cutoff_at=cutoff,
            model_version="lightgbm-0123456789ab",
        ),
    )
    fixture = serving_e2e_smoke._SmokeFixture(
        artifact_path=tmp_path / "artifact",
        history_start=history_start,
        history_end=cutoff + timedelta(hours=1),
        expected_cutoff=cutoff,
        expected_model_version="lightgbm-0123456789ab",
    )

    with pytest.raises(
        serving_e2e_smoke.ServingE2ESmokeError,
        match="seeded observations",
    ):
        serving_e2e_smoke._assert_serving_responses(
            client,
            fixture,
            "urbanflow_serving_e2e_test",
        )


@pytest.mark.parametrize("failure_mode", ["early_exit", "timeout"])
def test_startup_failures_raise_safe_error_with_only_a_bounded_log_tail(
    failure_mode: str,
) -> None:
    secret_url = "postgresql+psycopg://user:secret@localhost/urbanflow"
    artifact_path = Path("C:/private/smoke-artifact")
    log = io.BytesIO(
        (
            ("x" * (serving_e2e_smoke._MAX_LOG_TAIL_CHARS + 200))
            + f"\nurl={secret_url}\npassword=secret\nartifact={artifact_path}\n"
        ).encode()
    )
    process = SimpleNamespace(poll=(lambda: 7) if failure_mode == "early_exit" else (lambda: None))
    clock_values = iter([0.0, serving_e2e_smoke._STARTUP_TIMEOUT_SECONDS + 1.0])
    last_clock = 0.0

    def monotonic() -> float:
        nonlocal last_clock
        last_clock = next(clock_values, last_clock)
        return last_clock

    with pytest.raises(serving_e2e_smoke.ServingE2ESmokeError) as exc_info:
        serving_e2e_smoke._wait_for_api_ready(
            process,
            "http://127.0.0.1:45678/health",
            log,
            monotonic=monotonic,
            sleep=lambda _seconds: None,
            http_poller=lambda _url, _timeout: False,
            sensitive_values=(secret_url, str(artifact_path)),
        )

    message = str(exc_info.value)
    assert len(message) <= serving_e2e_smoke._MAX_LOG_TAIL_CHARS + 200
    assert "Log tail:" in message
    assert secret_url not in message
    assert "secret" not in message
    assert str(artifact_path) not in message


def test_log_tail_redacts_secret_split_across_truncation_boundary() -> None:
    secret_url = "postgresql+psycopg://user:boundary-secret@localhost/urbanflow"
    split_at = secret_url.index("boundary-secret")
    suffix_length = serving_e2e_smoke._MAX_LOG_TAIL_CHARS - (len(secret_url) - split_at)
    log = io.BytesIO(("prefix\n" + secret_url + ("z" * suffix_length)).encode())

    tail = serving_e2e_smoke._safe_log_tail(
        log,
        sensitive_values=(secret_url,),
    )

    assert len(tail) <= serving_e2e_smoke._MAX_LOG_TAIL_CHARS
    assert "boundary-secret" not in tail
    assert "localhost/urbanflow" not in tail


def test_log_tail_drops_unknown_credential_fragment_from_incomplete_first_line() -> None:
    credential_fragment = "unlisted-boundary-secret"
    credential_value = credential_fragment * 300
    log = io.BytesIO(f"password={credential_value}\nfinal safe log line\n".encode())

    tail = serving_e2e_smoke._safe_log_tail(
        log,
        sensitive_values=(),
    )

    assert len(tail) <= serving_e2e_smoke._MAX_LOG_TAIL_CHARS
    assert credential_fragment not in tail
    assert tail == "[truncated]\nfinal safe log line"


@pytest.mark.parametrize(
    ("stops_after_terminate", "expected_events"),
    [
        (True, ["terminate", ("wait", 10.0)]),
        (False, ["terminate", ("wait", 10.0), "kill", ("wait", 10.0)]),
    ],
)
def test_process_cleanup_terminates_then_bounded_waits_and_kills_only_if_needed(
    stops_after_terminate: bool,
    expected_events: list[object],
) -> None:
    events: list[object] = []

    class FakeProcess:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            events.append("terminate")

        def wait(self, timeout: float) -> int:
            events.append(("wait", timeout))
            if not stops_after_terminate and "kill" not in events:
                raise subprocess.TimeoutExpired("uvicorn", timeout)
            return 0

        def kill(self) -> None:
            events.append("kill")

    serving_e2e_smoke._stop_process(FakeProcess())

    assert events == expected_events


def test_http_assertion_failure_still_closes_and_cleans_every_created_resource() -> None:
    statements: list[str] = []
    process_events: list[object] = []

    class FakeConnection:
        def exec_driver_sql(self, statement: str) -> None:
            statements.append(statement)

    class FakeTransaction:
        def __enter__(self) -> FakeConnection:
            return FakeConnection()

        def __exit__(self, *_args: object) -> bool:
            return False

    class FakeEngine:
        disposed = False

        def begin(self) -> FakeTransaction:
            return FakeTransaction()

        def dispose(self) -> None:
            self.disposed = True

    class FakeProcess:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            process_events.append("terminate")

        def wait(self, timeout: float) -> int:
            process_events.append(("wait", timeout))
            return 0

        def kill(self) -> None:
            process_events.append("kill")

    class FakeClient:
        closed = False

        def close(self) -> None:
            self.closed = True

    fake_engine = FakeEngine()
    fake_process = FakeProcess()
    fake_client = FakeClient()
    fake_log = io.BytesIO()
    cutoff = datetime(2026, 7, 30, 2, tzinfo=UTC)
    temporary_paths: list[Path] = []
    captured_process_arguments: dict[str, Any] = {}

    def fixture_builder(
        _connection: object,
        temporary_directory: Path,
        _now: datetime,
    ) -> serving_e2e_smoke._SmokeFixture:
        temporary_paths.append(temporary_directory)
        artifact_path = temporary_directory / "artifact"
        artifact_path.mkdir()
        return serving_e2e_smoke._SmokeFixture(
            artifact_path=artifact_path,
            history_start=cutoff - timedelta(hours=191),
            history_end=cutoff + timedelta(hours=1),
            expected_cutoff=cutoff,
            expected_model_version="lightgbm-0123456789ab",
        )

    def process_factory(command: list[str], **kwargs: object) -> FakeProcess:
        captured_process_arguments["command"] = command
        captured_process_arguments.update(kwargs)
        return fake_process

    def fail_http_assertions(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "response exposed postgresql+psycopg://user:secret@localhost/urbanflow"
        )

    with pytest.raises(
        serving_e2e_smoke.ServingE2ESmokeError,
        match="Configured serving smoke failed",
    ) as exc_info:
        serving_e2e_smoke._run_serving_e2e_smoke(
            "postgresql+psycopg://user:secret@localhost/urbanflow?sslmode=require",
            schema_name="urbanflow_serving_e2e_test",
            engine_factory=lambda _url: fake_engine,
            create_tables=lambda _connection: None,
            fixture_builder=fixture_builder,
            process_factory=process_factory,
            monotonic=lambda: 0.0,
            sleep=lambda _seconds: None,
            free_port_allocator=lambda: 45678,
            http_poller=lambda _url, _timeout: True,
            client_factory=lambda _base_url: fake_client,
            response_assertion=fail_http_assertions,
            temporary_log_factory=lambda: fake_log,
        )

    assert "secret" not in str(exc_info.value)
    rendered_exception = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "secret" not in rendered_exception
    assert "postgresql" not in rendered_exception
    assert fake_client.closed is True
    assert fake_log.closed is True
    assert temporary_paths and all(not path.exists() for path in temporary_paths)
    assert process_events == ["terminate", ("wait", 10.0)]
    assert statements == [
        'CREATE SCHEMA "urbanflow_serving_e2e_test"',
        'SET search_path TO "urbanflow_serving_e2e_test"',
        'DROP SCHEMA IF EXISTS "urbanflow_serving_e2e_test" CASCADE',
    ]
    assert fake_engine.disposed is True
    assert captured_process_arguments["stdout"] is fake_log
    assert captured_process_arguments["stderr"] is fake_log
    assert captured_process_arguments["shell"] is False
    command = captured_process_arguments["command"]
    assert command[1:] == [
        "-m",
        "uvicorn",
        "urbanflow.api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "45678",
        "--workers",
        "1",
    ]
    child_environment = captured_process_arguments["env"]
    assert isinstance(child_environment, dict)
    assert set(key for key in child_environment if key.startswith("URBANFLOW_")) == {
        "URBANFLOW_DATABASE_URL",
        "URBANFLOW_API_MODEL_ARTIFACT_PATH",
        "URBANFLOW_API_MAX_DATA_AGE_HOURS",
    }
