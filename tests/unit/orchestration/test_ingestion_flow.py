from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from urbanflow.ingestion.hourly_count_pipeline import HourlyCountIngestionResult
from urbanflow.ingestion.hourly_counts import HourlyCountDateRange
from urbanflow.ingestion.sensor_location_pipeline import SensorLocationIngestionResult
from urbanflow.orchestration import ingestion_flow
from urbanflow.validation.reports import ValidationIssue, ValidationReport


def sensor_result(tmp_path):
    return SensorLocationIngestionResult(
        source_dataset="pedestrian-counting-system-sensor-locations",
        snapshot_dataset="sensor_locations",
        source_url="https://example.test/sensors",
        extracted_at=datetime(2026, 6, 30, 1, 0, tzinfo=UTC),
        source_total_count=2,
        record_count=2,
        snapshot_path=tmp_path / "raw" / "sensor_locations" / "records.json",
        manifest_path=tmp_path / "manifests" / "sensor_locations.json",
    )


def hourly_result(tmp_path, date_range):
    return HourlyCountIngestionResult(
        source_dataset="pedestrian-counting-system-monthly-counts-per-hour",
        snapshot_dataset="hourly_counts",
        source_url="https://example.test/hourly.csv",
        extracted_at=datetime(2026, 6, 30, 1, 5, tzinfo=UTC),
        date_range=date_range,
        source_total_count=3,
        record_count=3,
        selected_columns=("id", "location_id", "sensing_date"),
        snapshot_path=tmp_path / "raw" / "hourly_counts" / "records.csv",
        manifest_path=tmp_path / "manifests" / "hourly_counts.json",
    )


def validation_report(dataset, snapshot_path, *, passed=True, warning_count=0):
    errors = ()
    if not passed:
        errors = (
            ValidationIssue(
                code="INVALID_ROW",
                message="invalid row for test",
            ),
        )
    warnings = tuple(
        ValidationIssue(
            code=f"WARNING_{index}",
            message="warning for test",
        )
        for index in range(warning_count)
    )
    return ValidationReport(
        dataset=dataset,
        snapshot_path=str(snapshot_path),
        validated_at=datetime(2026, 6, 30, 2, 0, tzinfo=UTC),
        row_count=7,
        errors=errors,
        warnings=warnings,
    )


def test_date_range_options_require_bounded_hourly_range():
    with pytest.raises(ingestion_flow.IngestionFlowConfigError, match="provide --year"):
        ingestion_flow.date_range_from_options()


def test_date_range_options_rejects_year_mixed_with_dates():
    with pytest.raises(ingestion_flow.IngestionFlowConfigError, match="not both"):
        ingestion_flow.date_range_from_options(
            year=2025,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
        )


def test_run_ingestion_flow_validates_snapshots_without_database(monkeypatch, tmp_path):
    calls = []
    expected_date_range = HourlyCountDateRange(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
    )

    def fake_ingest_sensor_locations_task(**kwargs):
        calls.append(("sensor_ingest", kwargs["raw_root_dir"], kwargs["page_limit"]))
        return sensor_result(tmp_path)

    def fake_ingest_hourly_counts_task(**kwargs):
        calls.append(("hourly_ingest", kwargs["raw_root_dir"], kwargs["date_range"]))
        return hourly_result(tmp_path, kwargs["date_range"])

    def fake_validate_snapshot_task(dataset, snapshot_path, *, report_root_dir):
        calls.append(("validate", dataset, snapshot_path, report_root_dir))
        warning_count = 1 if dataset == "hourly_counts" else 0
        return validation_report(dataset, snapshot_path, warning_count=warning_count)

    def fail_load_snapshots_to_database_task(*args, **kwargs):
        raise AssertionError("database loading should be skipped")

    monkeypatch.setattr(
        ingestion_flow,
        "ingest_sensor_locations_task",
        fake_ingest_sensor_locations_task,
    )
    monkeypatch.setattr(
        ingestion_flow,
        "ingest_hourly_counts_task",
        fake_ingest_hourly_counts_task,
    )
    monkeypatch.setattr(
        ingestion_flow,
        "validate_snapshot_task",
        fake_validate_snapshot_task,
    )
    monkeypatch.setattr(
        ingestion_flow,
        "load_snapshots_to_database_task",
        fail_load_snapshots_to_database_task,
    )

    result = ingestion_flow.run_ingestion_flow.fn(
        raw_root_dir=tmp_path / "raw",
        manifest_root_dir=tmp_path / "manifests",
        report_root_dir=tmp_path / "reports",
        year=2025,
        page_limit=25,
    )

    assert result.sensor_locations.dataset == "sensor_locations"
    assert result.sensor_locations.record_count == 2
    assert result.sensor_locations.validation_passed is True
    assert result.sensor_locations.validation_warning_count == 0
    assert result.hourly_counts.dataset == "hourly_counts"
    assert result.hourly_counts.record_count == 3
    assert result.hourly_counts.validation_warning_count == 1
    assert result.database_loads == ()
    assert calls == [
        ("sensor_ingest", tmp_path / "raw", 25),
        ("hourly_ingest", tmp_path / "raw", expected_date_range),
        (
            "validate",
            "sensor_locations",
            tmp_path / "raw" / "sensor_locations" / "records.json",
            tmp_path / "reports",
        ),
        (
            "validate",
            "hourly_counts",
            tmp_path / "raw" / "hourly_counts" / "records.csv",
            tmp_path / "reports",
        ),
    ]


def test_run_ingestion_flow_stops_on_validation_failure(monkeypatch, tmp_path):
    def fake_ingest_sensor_locations_task(**kwargs):
        return sensor_result(tmp_path)

    def fake_ingest_hourly_counts_task(**kwargs):
        return hourly_result(tmp_path, kwargs["date_range"])

    def fake_validate_snapshot_task(dataset, snapshot_path, *, report_root_dir):
        return validation_report(dataset, snapshot_path, passed=dataset != "hourly_counts")

    monkeypatch.setattr(
        ingestion_flow,
        "ingest_sensor_locations_task",
        fake_ingest_sensor_locations_task,
    )
    monkeypatch.setattr(
        ingestion_flow,
        "ingest_hourly_counts_task",
        fake_ingest_hourly_counts_task,
    )
    monkeypatch.setattr(
        ingestion_flow,
        "validate_snapshot_task",
        fake_validate_snapshot_task,
    )

    with pytest.raises(ingestion_flow.IngestionFlowError, match="Validation failed"):
        ingestion_flow.run_ingestion_flow.fn(
            raw_root_dir=tmp_path / "raw",
            manifest_root_dir=tmp_path / "manifests",
            report_root_dir=tmp_path / "reports",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
        )
