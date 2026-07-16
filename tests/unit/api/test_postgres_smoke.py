import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy.exc import SQLAlchemyError

from urbanflow.api import postgres_smoke


def test_api_repository_smoke_cli_returns_two_when_database_url_is_missing(capsys) -> None:
    exit_code = postgres_smoke.main([], environ={})

    assert exit_code == 2
    assert postgres_smoke.SMOKE_DATABASE_URL_ENV_VAR in capsys.readouterr().err


def test_api_repository_smoke_cli_runs_with_explicit_database_url(monkeypatch, capsys) -> None:
    calls: dict[str, str | None] = {}

    def fake_run_postgres_api_repository_smoke(
        database_url: str, *, schema_name: str | None = None
    ):
        calls["database_url"] = database_url
        calls["schema_name"] = schema_name
        return postgres_smoke.PostgresApiRepositorySmokeResult(
            schema_name="urbanflow_api_smoke_test",
            all_sensor_location_ids=[999001, 999002],
            active_sensor_location_ids=[999001],
            history_count=1,
        )

    monkeypatch.setattr(
        postgres_smoke,
        "run_postgres_api_repository_smoke",
        fake_run_postgres_api_repository_smoke,
    )

    exit_code = postgres_smoke.main(
        [
            "--database-url",
            "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
            "--schema-name",
            "urbanflow_api_smoke_test",
        ],
        environ={},
    )

    assert exit_code == 0
    assert calls == {
        "database_url": "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
        "schema_name": "urbanflow_api_smoke_test",
    }
    assert json.loads(capsys.readouterr().out) == {
        "active_sensor_location_ids": [999001],
        "all_sensor_location_ids": [999001, 999002],
        "history_count": 1,
        "schema_name": "urbanflow_api_smoke_test",
    }


def test_api_repository_smoke_rejects_an_unsafe_schema_before_creating_an_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_urls: list[str] = []

    def record_engine_creation(database_url: str) -> object:
        engine_urls.append(database_url)
        return object()

    monkeypatch.setattr(postgres_smoke, "create_database_engine", record_engine_creation)

    with pytest.raises(ValueError, match="safe PostgreSQL identifier"):
        postgres_smoke.run_postgres_api_repository_smoke(
            "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
            schema_name="urbanflow_api_smoke;drop schema public",
        )

    assert engine_urls == []


def test_api_repository_smoke_does_not_drop_schema_when_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []
    disposed = False

    class FakeConnection:
        def exec_driver_sql(self, statement: str) -> None:
            statements.append(statement)
            if statement.startswith("CREATE SCHEMA"):
                raise SQLAlchemyError("schema already exists")

    class FakeTransaction:
        def __enter__(self) -> FakeConnection:
            return FakeConnection()

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
            return False

    class FakeEngine:
        def begin(self) -> FakeTransaction:
            return FakeTransaction()

        def dispose(self) -> None:
            nonlocal disposed
            disposed = True

    monkeypatch.setattr(postgres_smoke, "create_database_engine", lambda _url: FakeEngine())

    with pytest.raises(SQLAlchemyError):
        postgres_smoke.run_postgres_api_repository_smoke(
            "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
            schema_name="urbanflow_api_smoke_existing",
        )

    assert statements == ['CREATE SCHEMA "urbanflow_api_smoke_existing"']
    assert disposed is True


def test_api_repository_smoke_cli_returns_one_for_sqlalchemy_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    def fail_with_sqlalchemy_error(database_url: str, *, schema_name: str | None = None):
        raise SQLAlchemyError(f"could not connect to {database_url} using {schema_name}")

    monkeypatch.setattr(
        postgres_smoke,
        "run_postgres_api_repository_smoke",
        fail_with_sqlalchemy_error,
    )

    exit_code = postgres_smoke.main(
        ["--database-url", "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"],
        environ={},
    )

    assert exit_code == 1
    assert "PostgreSQL API repository smoke test failed" in capsys.readouterr().err


def test_postgres_api_repository_smoke_script_help() -> None:
    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [
            sys.executable,
            repository_root / "scripts" / "smoke_test_postgres_api.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Smoke test UrbanFlow AU PostgreSQL API repositories" in result.stdout
