import json
from datetime import UTC, datetime

import pytest

from urbanflow.validation.pipeline import ValidationPipelineError, validate_snapshot


def test_validate_snapshot_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValidationPipelineError, match="Unsupported dataset"):
        validate_snapshot("unknown", tmp_path / "records.json")


def test_validate_snapshot_returns_read_error_report_for_unreadable_snapshot(tmp_path):
    report = validate_snapshot(
        "sensor_locations",
        tmp_path / "missing.json",
        validated_at=datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
    )

    assert report.passed is False
    assert report.row_count == 0
    assert report.errors[0].code == "SNAPSHOT_READ_ERROR"


def test_validate_snapshot_writes_report_when_report_root_is_provided(tmp_path):
    snapshot_path = tmp_path / "records.json"
    snapshot_path.write_text(
        json.dumps(
            [
                {
                    "location_id": 1,
                    "sensor_description": "Bourke Street",
                    "sensor_name": "Sensor A",
                    "installation_date": None,
                    "status": "A",
                    "latitude": -37.81,
                    "longitude": 144.96,
                }
            ]
        ),
        encoding="utf-8",
    )

    report = validate_snapshot(
        "sensor_locations",
        snapshot_path,
        report_root=tmp_path / "reports",
        validated_at=datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
    )

    assert report.passed is True
    report_path = tmp_path / "reports" / "sensor_locations" / "20260627T120000Z.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
