from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from urbanflow.validation.reports import (
    ValidationIssue,
    ValidationMetric,
    ValidationReport,
    utc_now,
)
from urbanflow.validation.snapshot_readers import read_sensor_locations_snapshot

SENSOR_LOCATION_DATASET = "sensor_locations"


def _non_blank() -> pa.Check:
    return pa.Check(lambda series: series.astype(str).str.strip().ne(""))


SENSOR_LOCATION_SCHEMA = pa.DataFrameSchema(
    {
        "location_id": pa.Column(int, pa.Check(lambda series: series >= 1), coerce=True),
        "sensor_description": pa.Column(str, _non_blank(), coerce=True),
        "sensor_name": pa.Column(str, _non_blank(), coerce=True),
        "installation_date": pa.Column(str, nullable=True),
        "status": pa.Column(str, _non_blank(), coerce=True),
        "latitude": pa.Column(
            float,
            pa.Check(lambda series: (series >= -90) & (series <= 90)),
            coerce=True,
        ),
        "longitude": pa.Column(
            float,
            pa.Check(lambda series: (series >= -180) & (series <= 180)),
            coerce=True,
        ),
    },
    strict=False,
)


def _schema_errors(frame: pd.DataFrame) -> tuple[ValidationIssue, ...]:
    try:
        SENSOR_LOCATION_SCHEMA.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return (
            ValidationIssue(
                code="SCHEMA_INVALID",
                message=(
                    "Sensor-location schema validation failed: "
                    f"{len(exc.failure_cases)} failure cases"
                ),
            ),
        )
    return ()


def _duplicate_location_errors(frame: pd.DataFrame) -> tuple[ValidationIssue, ...]:
    if "location_id" not in frame.columns:
        return ()
    location_ids = pd.to_numeric(frame["location_id"], errors="coerce")
    duplicate_mask = location_ids.duplicated(keep=False) & location_ids.notna()
    if not duplicate_mask.any():
        return ()
    rows = tuple(int(index) for index in frame.index[duplicate_mask][:10])
    return (
        ValidationIssue(
            code="DUPLICATE_LOCATION_ID",
            message="location_id values must be unique within a sensor-location snapshot",
            column="location_id",
            rows=rows,
        ),
    )


def validate_sensor_locations_frame(
    frame: pd.DataFrame,
    *,
    snapshot_path: Path,
    validated_at: datetime | None = None,
) -> ValidationReport:
    errors = _schema_errors(frame) + _duplicate_location_errors(frame)
    status_distribution = (
        frame["status"].astype(str).value_counts(dropna=False).sort_index().astype(int).to_dict()
        if "status" in frame.columns
        else {}
    )
    null_installation_date_count = (
        int(frame["installation_date"].isna().sum()) if "installation_date" in frame.columns else 0
    )
    metrics = (
        ValidationMetric(name="sensor_count", value=int(len(frame))),
        ValidationMetric(name="null_installation_date_count", value=null_installation_date_count),
        ValidationMetric(name="status_distribution", value=status_distribution),
    )
    return ValidationReport(
        dataset=SENSOR_LOCATION_DATASET,
        snapshot_path=str(snapshot_path),
        validated_at=validated_at or utc_now(),
        row_count=int(len(frame)),
        errors=errors,
        metrics=metrics,
    )


def validate_sensor_locations_snapshot(
    snapshot_path: Path,
    validated_at: datetime | None = None,
) -> ValidationReport:
    frame = read_sensor_locations_snapshot(snapshot_path)
    return validate_sensor_locations_frame(
        frame,
        snapshot_path=snapshot_path,
        validated_at=validated_at,
    )
