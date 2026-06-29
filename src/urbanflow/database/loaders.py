from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from urbanflow.database.repositories import upsert_hourly_rows, upsert_sensor_rows
from urbanflow.database.time import melbourne_observed_at, parse_source_date
from urbanflow.validation.pipeline import validate_snapshot
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


def _ensure_validation_passed(dataset: str, snapshot_path: Path):
    report = validate_snapshot(dataset, snapshot_path)
    if not report.passed:
        codes = ", ".join(issue.code for issue in report.errors)
        raise DatabaseLoadError(f"Validation failed for {dataset}: {codes}")
    return report


def load_sensor_locations_snapshot(
    session: Session,
    snapshot_path: Path,
) -> DatabaseLoadResult:
    report = _ensure_validation_passed("sensor_locations", snapshot_path)
    rows = sensor_rows_from_snapshot(snapshot_path)
    row_count = upsert_sensor_rows(session, rows)
    return DatabaseLoadResult(
        dataset="sensor_locations",
        row_count=row_count,
        validation_warning_count=len(report.warnings),
    )


def load_hourly_counts_snapshot(session: Session, snapshot_path: Path) -> DatabaseLoadResult:
    report = _ensure_validation_passed("hourly_counts", snapshot_path)
    rows = hourly_count_rows_from_snapshot(snapshot_path)
    row_count = upsert_hourly_rows(session, rows)
    return DatabaseLoadResult(
        dataset="hourly_counts",
        row_count=row_count,
        validation_warning_count=len(report.warnings),
    )
