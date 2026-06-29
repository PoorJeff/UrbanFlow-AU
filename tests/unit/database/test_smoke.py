import json
from pathlib import Path

import pytest
from sqlalchemy.exc import SQLAlchemyError

from urbanflow.database import smoke


def test_smoke_cli_returns_two_when_database_url_missing(capsys) -> None:
    exit_code = smoke.main([], environ={})

    assert exit_code == 2
    assert smoke.SMOKE_DATABASE_URL_ENV_VAR in capsys.readouterr().err


def test_smoke_cli_runs_with_explicit_database_url(monkeypatch, capsys) -> None:
    calls = {}

    def fake_run_postgres_persistence_smoke(database_url, *, schema_name=None):
        calls["database_url"] = database_url
        calls["schema_name"] = schema_name
        return smoke.PostgresSmokeResult(
            schema_name="urbanflow_smoke_test",
            sensor_row_count=1,
            hourly_row_count=1,
        )

    monkeypatch.setattr(
        smoke,
        "run_postgres_persistence_smoke",
        fake_run_postgres_persistence_smoke,
    )

    exit_code = smoke.main(
        [
            "--database-url",
            "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
            "--schema-name",
            "urbanflow_smoke_test",
        ],
        environ={},
    )

    assert exit_code == 0
    assert calls == {
        "database_url": "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
        "schema_name": "urbanflow_smoke_test",
    }
    assert json.loads(capsys.readouterr().out) == {
        "hourly_row_count": 1,
        "schema_name": "urbanflow_smoke_test",
        "sensor_row_count": 1,
    }


@pytest.mark.parametrize(
    "schema_name",
    ["UrbanFlow", "urbanflow-smoke", "urbanflow_smoke;drop schema public"],
)
def test_smoke_schema_name_rejects_unsafe_identifiers(schema_name) -> None:
    with pytest.raises(ValueError, match="safe PostgreSQL identifier"):
        smoke.validate_smoke_schema_name(schema_name)


def test_smoke_does_not_drop_schema_when_schema_creation_fails(monkeypatch) -> None:
    statements = []
    disposed = False

    class FakeConnection:
        def exec_driver_sql(self, statement):
            statements.append(statement)
            if statement.startswith("CREATE SCHEMA"):
                raise SQLAlchemyError("schema already exists")

    class FakeTransaction:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeEngine:
        def begin(self):
            return FakeTransaction()

        def dispose(self):
            nonlocal disposed
            disposed = True

    monkeypatch.setattr(smoke, "create_database_engine", lambda database_url: FakeEngine())

    with pytest.raises(SQLAlchemyError):
        smoke.run_postgres_persistence_smoke(
            "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
            schema_name="urbanflow_smoke_existing",
        )

    assert statements == ['CREATE SCHEMA "urbanflow_smoke_existing"']
    assert disposed is True


def test_postgres_smoke_script_help() -> None:
    import subprocess
    import sys

    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [
            sys.executable,
            repository_root / "scripts" / "smoke_test_postgres_persistence.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Smoke test UrbanFlow AU PostgreSQL persistence" in result.stdout
