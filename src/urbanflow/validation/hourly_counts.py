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
from urbanflow.validation.snapshot_readers import read_hourly_counts_snapshot

HOURLY_COUNT_DATASET = "hourly_counts"


def _non_blank() -> pa.Check:
    return pa.Check(lambda series: series.astype(str).str.strip().ne(""))


HOURLY_COUNT_SCHEMA = pa.DataFrameSchema(
    {
        "id": pa.Column(str, _non_blank(), coerce=True),
        "location_id": pa.Column(int, pa.Check(lambda series: series >= 1), coerce=True),
        "sensing_date": pa.Column(pa.DateTime, coerce=True),
        "hourday": pa.Column(
            int,
            pa.Check(lambda series: (series >= 0) & (series <= 23)),
            coerce=True,
        ),
        "direction_1": pa.Column(int, pa.Check(lambda series: series >= 0), coerce=True),
        "direction_2": pa.Column(int, pa.Check(lambda series: series >= 0), coerce=True),
        "pedestriancount": pa.Column(int, pa.Check(lambda series: series >= 0), coerce=True),
        "sensor_name": pa.Column(str, _non_blank(), coerce=True),
        "location": pa.Column(str, _non_blank(), coerce=True),
    },
    strict=False,
)


def _schema_errors(frame: pd.DataFrame) -> tuple[ValidationIssue, ...]:
    try:
        HOURLY_COUNT_SCHEMA.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return (
            ValidationIssue(
                code="SCHEMA_INVALID",
                message=(
                    f"Hourly-count schema validation failed: {len(exc.failure_cases)} failure cases"
                ),
            ),
        )
    return ()


def _duplicate_id_errors(frame: pd.DataFrame) -> tuple[ValidationIssue, ...]:
    if "id" not in frame.columns:
        return ()
    duplicate_mask = frame["id"].astype(str).str.strip().duplicated(keep=False)
    if not duplicate_mask.any():
        return ()
    rows = tuple(int(index) for index in frame.index[duplicate_mask][:10])
    return (
        ValidationIssue(
            code="DUPLICATE_SOURCE_ID",
            message="id values must be unique within an hourly-count snapshot",
            column="id",
            rows=rows,
        ),
    )


def _direction_total_errors(frame: pd.DataFrame) -> tuple[ValidationIssue, ...]:
    required = {"direction_1", "direction_2", "pedestriancount"}
    if not required.issubset(frame.columns):
        return ()
    direction_1 = pd.to_numeric(frame["direction_1"], errors="coerce")
    direction_2 = pd.to_numeric(frame["direction_2"], errors="coerce")
    total = pd.to_numeric(frame["pedestriancount"], errors="coerce")
    comparable = direction_1.notna() & direction_2.notna() & total.notna()
    mismatch_mask = comparable & ((direction_1 + direction_2) != total)
    if not mismatch_mask.any():
        return ()
    rows = tuple(int(index) for index in frame.index[mismatch_mask][:10])
    return (
        ValidationIssue(
            code="DIRECTION_TOTAL_MISMATCH",
            message="direction_1 + direction_2 must equal pedestriancount",
            column="pedestriancount",
            rows=rows,
        ),
    )


def _diagnostic_warnings(frame: pd.DataFrame) -> tuple[ValidationIssue, ...]:
    warnings: list[ValidationIssue] = []
    key_columns = ["location_id", "sensing_date", "hourday"]
    if set(key_columns).issubset(frame.columns):
        duplicate_mask = frame.duplicated(subset=key_columns, keep=False)
        if duplicate_mask.any():
            warnings.append(
                ValidationIssue(
                    code="DUPLICATE_SENSOR_HOUR",
                    message="Duplicate location/date/hour keys need source investigation",
                    rows=tuple(int(index) for index in frame.index[duplicate_mask][:10]),
                )
            )
        typed = pd.DataFrame(
            {
                "location_id": pd.to_numeric(frame["location_id"], errors="coerce"),
                "sensing_date": pd.to_datetime(frame["sensing_date"], errors="coerce"),
                "hourday": pd.to_numeric(frame["hourday"], errors="coerce"),
            }
        ).dropna()
        if not typed.empty:
            coverage = typed.groupby(["location_id", "sensing_date"])["hourday"].nunique()
            incomplete_groups = int((coverage < 24).sum())
            if incomplete_groups:
                warnings.append(
                    ValidationIssue(
                        code="INCOMPLETE_HOUR_COVERAGE",
                        message=(
                            f"{incomplete_groups} location-date groups have fewer "
                            "than 24 observed hours"
                        ),
                    )
                )
    return tuple(warnings)


def _metrics(frame: pd.DataFrame) -> tuple[ValidationMetric, ...]:
    parsed_dates = (
        pd.to_datetime(frame["sensing_date"], errors="coerce")
        if "sensing_date" in frame.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    valid_dates = parsed_dates.dropna()
    date_range = (
        {
            "start": valid_dates.min().date().isoformat(),
            "end": valid_dates.max().date().isoformat(),
        }
        if not valid_dates.empty
        else {"start": None, "end": None}
    )
    hour_distribution = (
        pd.to_numeric(frame["hourday"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .astype(int)
        .rename(index=str)
        .to_dict()
        if "hourday" in frame.columns
        else {}
    )
    sensor_count = (
        int(pd.to_numeric(frame["location_id"], errors="coerce").dropna().nunique())
        if "location_id" in frame.columns
        else 0
    )
    return (
        ValidationMetric(name="row_count", value=int(len(frame))),
        ValidationMetric(name="sensor_count", value=sensor_count),
        ValidationMetric(name="date_range", value=date_range),
        ValidationMetric(name="hour_distribution", value=hour_distribution),
    )


def validate_hourly_counts_frame(
    frame: pd.DataFrame,
    *,
    snapshot_path: Path,
    validated_at: datetime | None = None,
) -> ValidationReport:
    errors = _schema_errors(frame) + _duplicate_id_errors(frame) + _direction_total_errors(frame)
    return ValidationReport(
        dataset=HOURLY_COUNT_DATASET,
        snapshot_path=str(snapshot_path),
        validated_at=validated_at or utc_now(),
        row_count=int(len(frame)),
        errors=errors,
        warnings=_diagnostic_warnings(frame),
        metrics=_metrics(frame),
    )


def validate_hourly_counts_snapshot(
    snapshot_path: Path,
    validated_at: datetime | None = None,
) -> ValidationReport:
    frame = read_hourly_counts_snapshot(snapshot_path)
    return validate_hourly_counts_frame(
        frame,
        snapshot_path=snapshot_path,
        validated_at=validated_at,
    )
