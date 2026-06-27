import json
from datetime import UTC, datetime

from urbanflow.validation.sensor_locations import validate_sensor_locations_snapshot


def _write_snapshot(tmp_path, records):
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _valid_record(**overrides):
    record = {
        "location_id": 1,
        "sensor_description": "Bourke Street",
        "sensor_name": "Sensor A",
        "installation_date": None,
        "status": "A",
        "latitude": -37.81,
        "longitude": 144.96,
    }
    record.update(overrides)
    return record


def test_sensor_location_snapshot_passes_and_records_metrics(tmp_path):
    snapshot_path = _write_snapshot(
        tmp_path,
        [
            _valid_record(location_id=1, status="A"),
            _valid_record(location_id=2, status="I", installation_date="2020-01-01"),
        ],
    )

    report = validate_sensor_locations_snapshot(
        snapshot_path,
        datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
    )

    assert report.passed is True
    assert report.row_count == 2
    assert report.to_dict()["metrics"]["sensor_count"] == 2
    assert report.to_dict()["metrics"]["null_installation_date_count"] == 1
    assert report.to_dict()["metrics"]["status_distribution"] == {"A": 1, "I": 1}


def test_sensor_location_snapshot_fails_for_duplicate_location_id(tmp_path):
    snapshot_path = _write_snapshot(tmp_path, [_valid_record(), _valid_record()])

    report = validate_sensor_locations_snapshot(snapshot_path)

    assert report.passed is False
    assert any(issue.code == "DUPLICATE_LOCATION_ID" for issue in report.errors)


def test_sensor_location_snapshot_fails_for_blank_name_and_bad_coordinates(tmp_path):
    snapshot_path = _write_snapshot(
        tmp_path,
        [_valid_record(sensor_name=" ", latitude=-100, longitude=200)],
    )

    report = validate_sensor_locations_snapshot(snapshot_path)

    assert report.passed is False
    assert any(issue.code == "SCHEMA_INVALID" for issue in report.errors)
