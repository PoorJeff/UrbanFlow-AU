from __future__ import annotations

import json
from pathlib import Path

from urbanflow.orchestration import cli
from urbanflow.orchestration.ingestion_flow import (
    DatabaseFlowResult,
    IngestionFlowResult,
    SnapshotFlowResult,
)


def flow_result() -> IngestionFlowResult:
    return IngestionFlowResult(
        sensor_locations=SnapshotFlowResult(
            dataset="sensor_locations",
            snapshot_path="data/raw/sensor_locations/records.json",
            manifest_path="data/manifests/sensor_locations.json",
            record_count=2,
            validation_passed=True,
            validation_error_count=0,
            validation_warning_count=0,
        ),
        hourly_counts=SnapshotFlowResult(
            dataset="hourly_counts",
            snapshot_path="data/raw/hourly_counts/records.csv",
            manifest_path="data/manifests/hourly_counts.json",
            record_count=3,
            validation_passed=True,
            validation_error_count=0,
            validation_warning_count=1,
        ),
        database_loads=(
            DatabaseFlowResult(
                dataset="sensor_locations",
                row_count=2,
                validation_warning_count=0,
            ),
        ),
    )


def test_ingestion_flow_cli_returns_two_when_date_range_missing(capsys):
    exit_code = cli.main([], environ={})

    assert exit_code == 2
    assert "provide --year" in capsys.readouterr().err


def test_ingestion_flow_cli_runs_flow_without_database(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_run_ingestion_flow(**kwargs):
        calls.update(kwargs)
        return flow_result()

    monkeypatch.setattr(cli, "run_ingestion_flow", fake_run_ingestion_flow)

    exit_code = cli.main(
        [
            "--raw-root",
            str(tmp_path / "raw"),
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--report-root",
            str(tmp_path / "reports"),
            "--year",
            "2025",
            "--page-limit",
            "25",
        ],
        environ={},
    )

    assert exit_code == 0
    assert calls["raw_root_dir"] == tmp_path / "raw"
    assert calls["manifest_root_dir"] == tmp_path / "manifests"
    assert calls["report_root_dir"] == tmp_path / "reports"
    assert calls["year"] == 2025
    assert calls["start_date"] is None
    assert calls["end_date"] is None
    assert calls["page_limit"] == 25
    assert calls["load_to_database"] is False
    assert calls["database_url"] is None
    assert json.loads(capsys.readouterr().out)["hourly_counts"]["record_count"] == 3


def test_ingestion_flow_cli_requires_database_url_when_loading(capsys):
    exit_code = cli.main(["--year", "2025", "--load-to-database"], environ={})

    assert exit_code == 2
    assert "Database URL is required" in capsys.readouterr().err


def test_ingestion_flow_cli_uses_database_url_from_environment(monkeypatch, capsys):
    calls = {}

    def fake_run_ingestion_flow(**kwargs):
        calls.update(kwargs)
        return flow_result()

    monkeypatch.setattr(cli, "run_ingestion_flow", fake_run_ingestion_flow)

    exit_code = cli.main(
        ["--year", "2025", "--load-to-database"],
        environ={"URBANFLOW_DATABASE_URL": "postgresql+psycopg://env"},
    )

    assert exit_code == 0
    assert calls["load_to_database"] is True
    assert calls["database_url"] == "postgresql+psycopg://env"
    assert json.loads(capsys.readouterr().out)["database_loads"][0]["row_count"] == 2


def test_ingestion_flow_script_help() -> None:
    import subprocess
    import sys

    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, repository_root / "scripts" / "run_ingestion_flow.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run the UrbanFlow AU Prefect ingestion flow" in result.stdout
