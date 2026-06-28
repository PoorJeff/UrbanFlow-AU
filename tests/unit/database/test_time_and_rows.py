import json
from datetime import date, datetime

from urbanflow.database.loaders import (
    hourly_count_rows_from_snapshot,
    sensor_rows_from_snapshot,
)
from urbanflow.database.time import melbourne_observed_at


def test_melbourne_observed_at_returns_timezone_aware_hour() -> None:
    observed_at = melbourne_observed_at("2025-01-01", "7")

    assert observed_at == datetime.fromisoformat("2025-01-01T07:00:00+11:00")
    assert observed_at.tzinfo is not None


def test_sensor_rows_from_snapshot_normalizes_database_shape(tmp_path) -> None:
    snapshot_path = tmp_path / "records.json"
    snapshot_path.write_text(
        json.dumps(
            [
                {
                    "location_id": 1,
                    "sensor_description": "Bourke Street",
                    "sensor_name": "Sensor A",
                    "installation_date": "2020-01-02",
                    "status": "A",
                    "latitude": -37.81,
                    "longitude": 144.96,
                },
                {
                    "location_id": 2,
                    "sensor_description": "Null Date",
                    "sensor_name": "Sensor B",
                    "installation_date": None,
                    "status": "I",
                    "latitude": -37.82,
                    "longitude": 144.97,
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = sensor_rows_from_snapshot(snapshot_path)

    assert rows == [
        {
            "location_id": 1,
            "sensor_name": "Sensor A",
            "sensor_description": "Bourke Street",
            "latitude": -37.81,
            "longitude": 144.96,
            "installation_date": date(2020, 1, 2),
            "status": "A",
        },
        {
            "location_id": 2,
            "sensor_name": "Sensor B",
            "sensor_description": "Null Date",
            "latitude": -37.82,
            "longitude": 144.97,
            "installation_date": None,
            "status": "I",
        },
    ]


def test_hourly_count_rows_from_snapshot_normalizes_database_shape(tmp_path) -> None:
    snapshot_path = tmp_path / "records.csv"
    snapshot_path.write_text(
        "id,location_id,sensing_date,hourday,direction_1,direction_2,"
        "pedestriancount,sensor_name,location\n"
        'abc,1,2025-01-01,7,2,3,5,Sensor A,"-37.81,144.96"\n',
        encoding="utf-8",
    )

    rows = hourly_count_rows_from_snapshot(snapshot_path)

    assert rows == [
        {
            "location_id": 1,
            "observed_at": datetime.fromisoformat("2025-01-01T07:00:00+11:00"),
            "source_sensing_date": date(2025, 1, 1),
            "source_hourday": 7,
            "pedestrian_count": 5,
            "direction_1_count": 2,
            "direction_2_count": 3,
            "source_snapshot_path": str(snapshot_path),
        }
    ]
