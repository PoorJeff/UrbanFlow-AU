import hashlib
import json
from datetime import UTC, datetime

import pytest

from urbanflow.ingestion.manifests import write_manifest
from urbanflow.ingestion.snapshots import format_extracted_at, write_json_snapshot

EXTRACTED_AT = datetime(2026, 6, 25, 8, 30, 5, tzinfo=UTC)
RECORDS = [
    {
        "location_id": 3,
        "sensor_name": "Swa295_T",
        "sensor_description": "Melbourne Central",
        "installation_date": "2009-03-25",
        "status": "A",
        "latitude": -37.81101524,
        "longitude": 144.96429485,
        "note": None,
        "location_type": "Outdoor",
        "direction_1": "North",
        "direction_2": "South",
        "location": {"lat": -37.81101524, "lon": 144.96429485},
    }
]


def test_format_extracted_at_uses_utc_compact_timestamp() -> None:
    assert format_extracted_at(EXTRACTED_AT) == "20260625T083005Z"


def test_write_json_snapshot_is_deterministic_and_immutable(tmp_path) -> None:
    snapshot_path = write_json_snapshot(
        records=RECORDS,
        root_dir=tmp_path,
        dataset="sensor_locations",
        extracted_at=EXTRACTED_AT,
    )

    assert snapshot_path.as_posix().endswith(
        "melbourne/sensor_locations/extracted_at=20260625T083005Z/records.json"
    )
    assert json.loads(snapshot_path.read_text(encoding="utf-8")) == RECORDS
    assert snapshot_path.read_text(encoding="utf-8").endswith("\n")

    with pytest.raises(FileExistsError):
        write_json_snapshot(
            records=RECORDS,
            root_dir=tmp_path,
            dataset="sensor_locations",
            extracted_at=EXTRACTED_AT,
        )


def test_write_manifest_records_snapshot_hash_and_counts(tmp_path) -> None:
    snapshot_path = write_json_snapshot(
        records=RECORDS,
        root_dir=tmp_path / "raw",
        dataset="sensor_locations",
        extracted_at=EXTRACTED_AT,
    )

    manifest_path = write_manifest(
        root_dir=tmp_path / "manifests",
        dataset="sensor_locations",
        source_url="https://example.test/datasets/sensor_locations/records",
        extracted_at=EXTRACTED_AT,
        record_count=1,
        source_total_count=136,
        snapshot_path=snapshot_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

    assert manifest["schema_version"] == 1
    assert manifest["dataset"] == "sensor_locations"
    assert manifest["record_count"] == 1
    assert manifest["source_total_count"] == 136
    assert manifest["snapshot_sha256"] == expected_hash
    assert manifest["snapshot_path"] == snapshot_path.as_posix()
