from __future__ import annotations

from datetime import datetime
from pathlib import Path

from urbanflow.validation.reports import ValidationIssue, ValidationReport, utc_now
from urbanflow.validation.snapshot_readers import SnapshotReadError


class ValidationPipelineError(Exception):
    """Raised when validation cannot be routed or configured."""


def _report_path(report_root: Path, dataset: str, validated_at: datetime) -> Path:
    timestamp = validated_at.strftime("%Y%m%dT%H%M%SZ")
    return report_root / dataset / f"{timestamp}.json"


def _read_error_report(
    dataset: str,
    snapshot_path: Path,
    exc: SnapshotReadError,
    validated_at: datetime,
) -> ValidationReport:
    return ValidationReport(
        dataset=dataset,
        snapshot_path=str(snapshot_path),
        validated_at=validated_at,
        row_count=0,
        errors=(
            ValidationIssue(
                code="SNAPSHOT_READ_ERROR",
                message=str(exc),
            ),
        ),
    )


def _validate_dataset(
    dataset: str,
    snapshot_path: Path,
    validated_at: datetime,
) -> ValidationReport:
    if dataset == "sensor_locations":
        from urbanflow.validation.sensor_locations import validate_sensor_locations_snapshot

        return validate_sensor_locations_snapshot(snapshot_path, validated_at)
    if dataset == "hourly_counts":
        from urbanflow.validation.hourly_counts import validate_hourly_counts_snapshot

        return validate_hourly_counts_snapshot(snapshot_path, validated_at)
    raise ValidationPipelineError(f"Unsupported dataset: {dataset}")


def validate_snapshot(
    dataset: str,
    snapshot_path: Path,
    *,
    report_root: Path | None = None,
    validated_at: datetime | None = None,
) -> ValidationReport:
    if dataset not in {"sensor_locations", "hourly_counts"}:
        raise ValidationPipelineError(f"Unsupported dataset: {dataset}")

    timestamp = validated_at or utc_now()
    try:
        report = _validate_dataset(dataset, snapshot_path, timestamp)
    except SnapshotReadError as exc:
        report = _read_error_report(dataset, snapshot_path, exc, timestamp)

    if report_root is not None:
        report.write_json(_report_path(report_root, dataset, report.validated_at))
    return report
