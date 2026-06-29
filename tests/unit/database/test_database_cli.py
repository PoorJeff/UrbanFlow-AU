import json
from pathlib import Path

from urbanflow.database.cli import main
from urbanflow.database.loaders import DatabaseLoadResult


def test_database_cli_returns_two_when_database_url_missing(tmp_path, capsys) -> None:
    exit_code = main(["sensor_locations", str(tmp_path / "records.json")], environ={})

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Database URL is required" in captured.err


def test_database_cli_loads_sensor_snapshot(monkeypatch, tmp_path, capsys) -> None:
    snapshot_path = tmp_path / "records.json"
    calls = {}

    class FakeSessionFactory:
        def begin(self):
            class Context:
                def __enter__(self):
                    return "session"

                def __exit__(self, exc_type, exc, tb):
                    return False

            return Context()

    def fake_load_sensor_locations_snapshot(session, path):
        calls["loaded"] = (session, path)
        return DatabaseLoadResult(
            dataset="sensor_locations",
            row_count=3,
            validation_warning_count=0,
        )

    monkeypatch.setattr(
        "urbanflow.database.cli.create_database_engine",
        lambda url: calls.setdefault("url", url),
    )
    monkeypatch.setattr(
        "urbanflow.database.cli.create_session_factory",
        lambda engine: FakeSessionFactory(),
    )
    monkeypatch.setattr(
        "urbanflow.database.cli.load_sensor_locations_snapshot",
        fake_load_sensor_locations_snapshot,
    )

    exit_code = main(
        ["sensor_locations", str(snapshot_path), "--database-url", "postgresql+psycopg://db"],
        environ={},
    )

    assert exit_code == 0
    assert calls["url"] == "postgresql+psycopg://db"
    assert calls["loaded"] == ("session", snapshot_path)
    assert json.loads(capsys.readouterr().out)["dataset"] == "sensor_locations"


def test_database_load_script_help() -> None:
    import subprocess
    import sys

    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, repository_root / "scripts" / "load_snapshot_to_db.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Load a validated UrbanFlow AU snapshot into PostgreSQL" in result.stdout
