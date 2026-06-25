from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUIRED_FIELDS = (
    "location_id",
    "sensor_description",
    "sensor_name",
    "installation_date",
    "status",
    "latitude",
    "longitude",
)


class SensorLocationParseError(ValueError):
    """Raised when a source sensor-location record cannot be normalized."""


@dataclass(frozen=True)
class SensorLocation:
    location_id: int
    sensor_description: str
    sensor_name: str
    installation_date: str
    status: str
    latitude: float
    longitude: float
    note: str | None = None
    location_type: str | None = None
    direction_1: str | None = None
    direction_2: str | None = None
    location: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "location_id": self.location_id,
            "sensor_description": self.sensor_description,
            "sensor_name": self.sensor_name,
            "installation_date": self.installation_date,
            "status": self.status,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "note": self.note,
            "location_type": self.location_type,
            "direction_1": self.direction_1,
            "direction_2": self.direction_2,
            "location": self.location,
        }


def parse_sensor_location(record: dict[str, Any]) -> SensorLocation:
    for field in REQUIRED_FIELDS:
        if field not in record or record[field] is None:
            raise SensorLocationParseError(f"Sensor location record is missing required field '{field}'")

    location_id = _coerce_int(record["location_id"], "location_id")
    latitude = _coerce_float(record["latitude"], "latitude")
    longitude = _coerce_float(record["longitude"], "longitude")
    _validate_coordinates(latitude=latitude, longitude=longitude)

    return SensorLocation(
        location_id=location_id,
        sensor_description=str(record["sensor_description"]),
        sensor_name=str(record["sensor_name"]),
        installation_date=str(record["installation_date"]),
        status=str(record["status"]),
        latitude=latitude,
        longitude=longitude,
        note=_optional_str(record.get("note")),
        location_type=_optional_str(record.get("location_type")),
        direction_1=_optional_str(record.get("direction_1")),
        direction_2=_optional_str(record.get("direction_2")),
        location=dict(record["location"]) if isinstance(record.get("location"), dict) else None,
    )


def normalize_sensor_locations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [parse_sensor_location(record).to_dict() for record in records]


def _coerce_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SensorLocationParseError(f"Field '{field}' must be an integer") from exc


def _coerce_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SensorLocationParseError(f"Field '{field}' must be numeric") from exc


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _validate_coordinates(*, latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise SensorLocationParseError("Field 'latitude' must be between -90 and 90")
    if not -180 <= longitude <= 180:
        raise SensorLocationParseError("Field 'longitude' must be between -180 and 180")
