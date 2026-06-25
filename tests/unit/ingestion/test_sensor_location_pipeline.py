import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from urbanflow.ingestion.melbourne_api import DatasetRecords
from urbanflow.ingestion.sensor_location_pipeline import (
    SENSOR_LOCATIONS_SOURCE_DATASET,
    SensorLocationIngestionResult,
    ingest_sensor_locations,
)
from urbanflow.ingestion.sensor_locations import SensorLocationParseError

EXTRACTED_AT = datetime(2026, 6, 25, 9, 45, 0, tzinfo=UTC)
SOURCE_RECORD = {
    "location_id": 3,
    "sensor_description": "Melbourne Central",
    "sensor_name": "Swa295_T",
    "installation_date": "2009-03-25",
    "note": None,
    "location_type": "Outdoor",
    "status": "A",
    "direction_1": "North",
    "direction_2": "South",
    "latitude": -37.81101524,
    "longitude": 144.96429485,
    "location": {"lon": 144.96429485, "lat": -37.81101524},
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


def test_ingest_sensor_locations_writes_snapshot_manifest_and_returns_metadata(
    tmp_path: Path,
) -> None:
    api_client = FakeApiClient([SOURCE_RECORD], total_count=136)

    result = ingest_sensor_locations(
        api_client=api_client,
        raw_root_dir=tmp_path / "raw",
        manifest_root_dir=tmp_path / "manifests",
        extracted_at=EXTRACTED_AT,
        page_limit=50,
    )

    assert isinstance(result, SensorLocationIngestionResult)
    assert api_client.calls == [(SENSOR_LOCATIONS_SOURCE_DATASET, 50)]
    assert result.source_total_count == 136
    assert result.record_count == 1
    assert result.snapshot_path.exists()
    assert result.manifest_path.exists()

    snapshot = json.loads(result.snapshot_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert snapshot[0]["location_id"] == 3
    assert snapshot[0]["sensor_name"] == "Swa295_T"
    assert manifest["record_count"] == 1
    assert manifest["source_total_count"] == 136
    assert manifest["source_url"] == result.source_url
    assert manifest["snapshot_path"] == result.snapshot_path.as_posix()


def test_ingest_sensor_locations_fails_before_writing_outputs_for_invalid_record(
    tmp_path: Path,
) -> None:
    invalid_record = dict(SOURCE_RECORD)
    invalid_record["latitude"] = -120
    api_client = FakeApiClient([invalid_record])

    with pytest.raises(SensorLocationParseError):
        ingest_sensor_locations(
            api_client=api_client,
            raw_root_dir=tmp_path / "raw",
            manifest_root_dir=tmp_path / "manifests",
            extracted_at=EXTRACTED_AT,
        )

    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "manifests").exists()
