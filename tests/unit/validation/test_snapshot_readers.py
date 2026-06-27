import json

import pytest

from urbanflow.validation.snapshot_readers import (
    SnapshotReadError,
    read_hourly_counts_snapshot,
    read_sensor_locations_snapshot,
)


def test_read_sensor_locations_snapshot_loads_json_records(tmp_path):
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

    frame = read_sensor_locations_snapshot(snapshot_path)

    assert frame.to_dict("records")[0]["sensor_name"] == "Sensor A"


def test_read_sensor_locations_snapshot_rejects_non_list_json(tmp_path):
    snapshot_path = tmp_path / "records.json"
    snapshot_path.write_text(json.dumps({"records": []}), encoding="utf-8")

    with pytest.raises(SnapshotReadError, match="JSON snapshot must contain a list"):
        read_sensor_locations_snapshot(snapshot_path)


def test_read_hourly_counts_snapshot_loads_csv_as_strings(tmp_path):
    snapshot_path = tmp_path / "records.csv"
    snapshot_path.write_text(
        "id,location_id,sensing_date,hourday,direction_1,direction_2,pedestriancount,sensor_name,location\n"
        'abc,1,2025-01-01,0,2,3,5,Sensor A,"-37.81,144.96"\n',
        encoding="utf-8",
    )

    frame = read_hourly_counts_snapshot(snapshot_path)

    assert frame.loc[0, "hourday"] == "0"
    assert frame.loc[0, "pedestriancount"] == "5"


def test_snapshot_reader_reports_missing_file(tmp_path):
    with pytest.raises(SnapshotReadError, match="Snapshot file does not exist"):
        read_hourly_counts_snapshot(tmp_path / "missing.csv")
