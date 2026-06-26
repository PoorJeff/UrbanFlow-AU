import json
from pathlib import Path
from typing import Any

import pytest

from urbanflow.ingestion.melbourne_api import DatasetRecords
from urbanflow.ingestion.sensor_location_cli import main

SOURCE_RECORD = {
    "location_id": 3,
    "sensor_description": "Melbourne Central",
    "sensor_name": "Swa295_T",
    "installation_date": "2009-03-25",
    "status": "A",
    "latitude": -37.81101524,
    "longitude": 144.96429485,
}


class FakeApiClient:
    def __init__(self, records: list[dict[str, Any]], *, total_count: int = 136) -> None:
        self.records = records
        self.total_count = total_count
        self.calls: list[tuple[str, int]] = []

    def fetch_all_records(self, dataset: str, *, limit: int = 100) -> DatasetRecords:
        self.calls.append((dataset, limit))
        return DatasetRecords(
            dataset=dataset,
            source_url=f"https://example.test/{dataset}/records",
            total_count=self.total_count,
            records=self.records,
        )


def test_main_uses_default_roots_and_prints_json_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [],
        api_client_factory=lambda http_client: FakeApiClient([SOURCE_RECORD]),
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["record_count"] == 1
    assert summary["source_total_count"] == 136
    assert (tmp_path / "data" / "raw").exists()
    assert (tmp_path / "data" / "manifests").exists()


def test_main_accepts_output_root_and_page_limit_overrides(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_root = tmp_path / "custom-raw"
    manifest_root = tmp_path / "custom-manifests"
    fake_client = FakeApiClient([SOURCE_RECORD])

    exit_code = main(
        [
            "--raw-root",
            str(raw_root),
            "--manifest-root",
            str(manifest_root),
            "--page-limit",
            "50",
        ],
        api_client_factory=lambda http_client: fake_client,
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert fake_client.calls == [("pedestrian-counting-system-sensor-locations", 50)]
    assert Path(summary["snapshot_path"]).is_relative_to(raw_root)
    assert Path(summary["manifest_path"]).is_relative_to(manifest_root)


def test_main_reports_expected_pipeline_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_record = {**SOURCE_RECORD, "latitude": -120}

    exit_code = main(
        [
            "--raw-root",
            str(tmp_path / "raw"),
            "--manifest-root",
            str(tmp_path / "manifests"),
        ],
        api_client_factory=lambda http_client: FakeApiClient([invalid_record]),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error: Field 'latitude' must be between -90 and 90" in captured.err
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "manifests").exists()


def test_main_rejects_non_positive_page_limit(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--page-limit", "0"])

    assert exc_info.value.code == 2
    assert "page limit must be greater than zero" in capsys.readouterr().err
