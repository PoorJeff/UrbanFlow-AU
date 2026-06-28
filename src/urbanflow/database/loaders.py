from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from urbanflow.database.time import melbourne_observed_at, parse_source_date
from urbanflow.validation.snapshot_readers import (
    read_hourly_counts_snapshot,
    read_sensor_locations_snapshot,
)


class DatabaseLoadError(Exception):
    """Raised when a validated snapshot cannot be loaded into the database."""


@dataclass(frozen=True)
class DatabaseLoadResult:
    dataset: str
    row_count: int
    validation_warning_count: int


def _parse_optional_date(value: Any) -> date | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return parse_source_date(value)


def sensor_rows_from_snapshot(snapshot_path: Path) -> list[dict[str, object]]:
    frame = read_sensor_locations_snapshot(snapshot_path)
    rows: list[dict[str, object]] = []
    for record in frame.to_dict("records"):
        rows.append(
            {
                "location_id": int(record["location_id"]),
                "sensor_name": str(record["sensor_name"]),
                "sensor_description": str(record["sensor_description"]),
                "latitude": float(record["latitude"]),
                "longitude": float(record["longitude"]),
                "installation_date": _parse_optional_date(record.get("installation_date")),
                "status": str(record["status"]),
            }
        )
    return rows


def hourly_count_rows_from_snapshot(snapshot_path: Path) -> list[dict[str, object]]:
    frame = read_hourly_counts_snapshot(snapshot_path)
    rows: list[dict[str, object]] = []
    for record in frame.to_dict("records"):
        source_date = parse_source_date(record["sensing_date"])
        source_hour = int(record["hourday"])
        rows.append(
            {
                "location_id": int(record["location_id"]),
                "observed_at": melbourne_observed_at(source_date, source_hour),
                "source_sensing_date": source_date,
                "source_hourday": source_hour,
                "pedestrian_count": int(record["pedestriancount"]),
                "direction_1_count": int(record["direction_1"]),
                "direction_2_count": int(record["direction_2"]),
                "source_snapshot_path": str(snapshot_path),
            }
        )
    return rows
