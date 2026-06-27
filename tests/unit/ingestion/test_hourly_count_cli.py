import json
import subprocess
import sys
from pathlib import Path

import pytest

from urbanflow.ingestion.hourly_count_cli import main
from urbanflow.ingestion.melbourne_api import DatasetRecordCount

CSV_BYTES = (
    b"id,location_id,sensing_date,hourday,direction_1,direction_2,"
    b"pedestriancount,sensor_name,location\n"
    b'51120250101,51,2025-01-01,1,100,79,179,Fra118_T,"-37.8, 144.9"\n'
)


class FakeHourlyApiClient:
    def count_records(self, dataset: str, *, where: str | None = None) -> DatasetRecordCount:
        return DatasetRecordCount(
            dataset=dataset,
            source_url=f"https://example.test/{dataset}/records",
            total_count=1,
        )

    def export_url(self, dataset: str, *, export_format: str) -> str:
        return f"https://example.test/{dataset}/exports/{export_format}"

    def export_csv(
        self,
        dataset: str,
        *,
        output_path: Path,
        select: tuple[str, ...],
        where: str | None = None,
    ) -> None:
        output_path.write_bytes(CSV_BYTES)


def test_main_accepts_year_and_prints_json_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        ["--year", "2025"],
        api_client_factory=lambda http_client: FakeHourlyApiClient(),
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["date_range"] == {"end": "2025-12-31", "start": "2025-01-01"}
    assert summary["record_count"] == 1
    assert summary["source_total_count"] == 1
    assert (tmp_path / "data" / "raw").exists()
    assert (tmp_path / "data" / "manifests").exists()


def test_main_accepts_explicit_date_range(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"

    exit_code = main(
        [
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-01",
            "--raw-root",
            str(raw_root),
            "--manifest-root",
            str(manifest_root),
        ],
        api_client_factory=lambda http_client: FakeHourlyApiClient(),
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["date_range"] == {"end": "2025-01-01", "start": "2025-01-01"}
    assert Path(summary["snapshot_path"]).is_relative_to(raw_root)
    assert Path(summary["manifest_path"]).is_relative_to(manifest_root)


def test_main_rejects_missing_end_date(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--start-date", "2025-01-01"])

    assert exc_info.value.code == 2
    assert "provide both --start-date and --end-date" in capsys.readouterr().err


def test_main_rejects_invalid_date_format(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--start-date", "2025/01/01", "--end-date", "2025-01-01"])

    assert exc_info.value.code == 2
    assert "YYYY-MM-DD" in capsys.readouterr().err


def test_main_reports_pipeline_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class EmptyApiClient(FakeHourlyApiClient):
        def count_records(
            self,
            dataset: str,
            *,
            where: str | None = None,
        ) -> DatasetRecordCount:
            return DatasetRecordCount(
                dataset=dataset,
                source_url=f"https://example.test/{dataset}/records",
                total_count=0,
            )

    exit_code = main(
        [
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-01",
            "--raw-root",
            str(tmp_path / "raw"),
            "--manifest-root",
            str(tmp_path / "manifests"),
        ],
        api_client_factory=lambda http_client: EmptyApiClient(),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error: No hourly-count rows found" in captured.err


def test_runner_script_displays_help() -> None:
    repository_root = Path(__file__).parents[3]

    completed_process = subprocess.run(
        [sys.executable, repository_root / "scripts" / "ingest_hourly_counts.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0
    assert "--year" in completed_process.stdout
    assert "--start-date" in completed_process.stdout
    assert "--end-date" in completed_process.stdout
