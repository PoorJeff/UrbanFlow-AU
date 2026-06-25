import pytest

from urbanflow.ingestion.sensor_locations import (
    SensorLocationParseError,
    normalize_sensor_locations,
    parse_sensor_location,
)

SOURCE_RECORD = {
    "location_id": "3",
    "sensor_description": "Melbourne Central",
    "sensor_name": "Swa295_T",
    "installation_date": "2009-03-25",
    "note": None,
    "location_type": "Outdoor",
    "status": "A",
    "direction_1": "North",
    "direction_2": "South",
    "latitude": "-37.81101524",
    "longitude": "144.96429485",
    "location": {"lon": 144.96429485, "lat": -37.81101524},
}


def test_parse_sensor_location_normalizes_required_and_optional_fields() -> None:
    sensor = parse_sensor_location(SOURCE_RECORD)

    assert sensor.to_dict() == {
        "location_id": 3,
        "sensor_description": "Melbourne Central",
        "sensor_name": "Swa295_T",
        "installation_date": "2009-03-25",
        "status": "A",
        "latitude": -37.81101524,
        "longitude": 144.96429485,
        "note": None,
        "location_type": "Outdoor",
        "direction_1": "North",
        "direction_2": "South",
        "location": {"lon": 144.96429485, "lat": -37.81101524},
    }


def test_parse_sensor_location_rejects_missing_required_field() -> None:
    record = dict(SOURCE_RECORD)
    record.pop("sensor_name")

    with pytest.raises(SensorLocationParseError, match="sensor_name"):
        parse_sensor_location(record)


def test_parse_sensor_location_rejects_invalid_coordinates() -> None:
    record = dict(SOURCE_RECORD)
    record["latitude"] = -120

    with pytest.raises(SensorLocationParseError, match="latitude"):
        parse_sensor_location(record)


def test_normalize_sensor_locations_returns_json_ready_dicts() -> None:
    records = normalize_sensor_locations([SOURCE_RECORD])

    assert records == [parse_sensor_location(SOURCE_RECORD).to_dict()]
