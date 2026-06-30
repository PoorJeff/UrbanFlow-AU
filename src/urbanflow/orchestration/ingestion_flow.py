from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TypeAlias

import httpx
from prefect import flow, task

from urbanflow.database.engine import create_database_engine, create_session_factory
from urbanflow.database.loaders import (
    DatabaseLoadResult,
    load_hourly_counts_snapshot,
    load_sensor_locations_snapshot,
)
from urbanflow.ingestion.hourly_count_pipeline import (
    HourlyCountIngestionResult,
    SupportsHourlyCountExport,
    ingest_hourly_counts,
)
from urbanflow.ingestion.hourly_counts import (
    HourlyCountDateRange,
    validate_date_range,
    year_date_range,
)
from urbanflow.ingestion.melbourne_api import MelbourneApiClient
from urbanflow.ingestion.sensor_location_pipeline import (
    SensorLocationIngestionResult,
    SupportsDatasetRecords,
    ingest_sensor_locations,
)
from urbanflow.validation.pipeline import validate_snapshot
from urbanflow.validation.reports import ValidationReport

SensorApiClientFactory: TypeAlias = Callable[[httpx.Client], SupportsDatasetRecords]
HourlyApiClientFactory: TypeAlias = Callable[[httpx.Client], SupportsHourlyCountExport]


class IngestionFlowConfigError(ValueError):
    """Raised when local flow configuration is incomplete or contradictory."""


class IngestionFlowError(RuntimeError):
    """Raised when the orchestration flow completes a step with failing data quality."""


@dataclass(frozen=True)
class SnapshotFlowResult:
    dataset: str
    snapshot_path: str
    manifest_path: str
    record_count: int
    validation_passed: bool
    validation_error_count: int
    validation_warning_count: int


@dataclass(frozen=True)
class DatabaseFlowResult:
    dataset: str
    row_count: int
    validation_warning_count: int


@dataclass(frozen=True)
class IngestionFlowResult:
    sensor_locations: SnapshotFlowResult
    hourly_counts: SnapshotFlowResult
    database_loads: tuple[DatabaseFlowResult, ...] = ()


def _default_sensor_api_client_factory(http_client: httpx.Client) -> MelbourneApiClient:
    return MelbourneApiClient(http_client=http_client)


def _default_hourly_api_client_factory(http_client: httpx.Client) -> MelbourneApiClient:
    return MelbourneApiClient(http_client=http_client)


def date_range_from_options(
    *,
    year: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> HourlyCountDateRange:
    has_year = year is not None
    has_start = start_date is not None
    has_end = end_date is not None
    if has_year and (has_start or has_end):
        raise IngestionFlowConfigError("provide either --year or --start-date/--end-date, not both")
    if has_year:
        return year_date_range(year)
    if has_start != has_end:
        raise IngestionFlowConfigError("provide both --start-date and --end-date")
    if not has_start or start_date is None or end_date is None:
        raise IngestionFlowConfigError("provide --year or both --start-date and --end-date")
    return validate_date_range(start_date, end_date)


def _ensure_report_passed(report: ValidationReport) -> None:
    if not report.passed:
        codes = ", ".join(issue.code for issue in report.errors)
        raise IngestionFlowError(f"Validation failed for {report.dataset}: {codes}")


def _snapshot_result(
    *,
    dataset: str,
    snapshot_path: Path,
    manifest_path: Path,
    record_count: int,
    report: ValidationReport,
) -> SnapshotFlowResult:
    _ensure_report_passed(report)
    return SnapshotFlowResult(
        dataset=dataset,
        snapshot_path=snapshot_path.as_posix(),
        manifest_path=manifest_path.as_posix(),
        record_count=record_count,
        validation_passed=report.passed,
        validation_error_count=len(report.errors),
        validation_warning_count=len(report.warnings),
    )


def _database_result(result: DatabaseLoadResult) -> DatabaseFlowResult:
    return DatabaseFlowResult(
        dataset=result.dataset,
        row_count=result.row_count,
        validation_warning_count=result.validation_warning_count,
    )


def load_snapshots_to_database(
    *,
    database_url: str,
    sensor_snapshot_path: Path,
    hourly_snapshot_path: Path,
) -> tuple[DatabaseFlowResult, ...]:
    engine = create_database_engine(database_url)
    try:
        session_factory = create_session_factory(engine)
        with session_factory.begin() as session:
            sensor_result = load_sensor_locations_snapshot(session, sensor_snapshot_path)
            hourly_result = load_hourly_counts_snapshot(session, hourly_snapshot_path)
    finally:
        engine.dispose()
    return (_database_result(sensor_result), _database_result(hourly_result))


@task(name="ingest-sensor-locations")
def ingest_sensor_locations_task(
    *,
    raw_root_dir: Path,
    manifest_root_dir: Path,
    page_limit: int,
    api_client_factory: SensorApiClientFactory | None = None,
) -> SensorLocationIngestionResult:
    factory = api_client_factory or _default_sensor_api_client_factory
    with httpx.Client(timeout=30.0) as http_client:
        return ingest_sensor_locations(
            api_client=factory(http_client),
            raw_root_dir=raw_root_dir,
            manifest_root_dir=manifest_root_dir,
            page_limit=page_limit,
        )


@task(name="ingest-hourly-counts")
def ingest_hourly_counts_task(
    *,
    raw_root_dir: Path,
    manifest_root_dir: Path,
    date_range: HourlyCountDateRange,
    api_client_factory: HourlyApiClientFactory | None = None,
) -> HourlyCountIngestionResult:
    factory = api_client_factory or _default_hourly_api_client_factory
    with httpx.Client(timeout=30.0) as http_client:
        return ingest_hourly_counts(
            api_client=factory(http_client),
            raw_root_dir=raw_root_dir,
            manifest_root_dir=manifest_root_dir,
            date_range=date_range,
        )


@task(name="validate-snapshot")
def validate_snapshot_task(
    dataset: str,
    snapshot_path: Path,
    *,
    report_root_dir: Path | None,
) -> ValidationReport:
    report = validate_snapshot(
        dataset,
        snapshot_path,
        report_root=report_root_dir,
    )
    _ensure_report_passed(report)
    return report


@task(name="load-snapshots-to-database")
def load_snapshots_to_database_task(
    *,
    database_url: str,
    sensor_snapshot_path: Path,
    hourly_snapshot_path: Path,
) -> tuple[DatabaseFlowResult, ...]:
    return load_snapshots_to_database(
        database_url=database_url,
        sensor_snapshot_path=sensor_snapshot_path,
        hourly_snapshot_path=hourly_snapshot_path,
    )


@flow(name="urbanflow-ingestion")
def run_ingestion_flow(
    *,
    raw_root_dir: Path = Path("data/raw"),
    manifest_root_dir: Path = Path("data/manifests"),
    report_root_dir: Path | None = Path("reports/data_quality"),
    year: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    page_limit: int = 100,
    load_to_database: bool = False,
    database_url: str | None = None,
    sensor_api_client_factory: SensorApiClientFactory | None = None,
    hourly_api_client_factory: HourlyApiClientFactory | None = None,
) -> IngestionFlowResult:
    if load_to_database and not database_url:
        raise IngestionFlowConfigError("database_url is required when load_to_database is true")

    date_range = date_range_from_options(
        year=year,
        start_date=start_date,
        end_date=end_date,
    )

    sensor_ingestion = ingest_sensor_locations_task(
        raw_root_dir=raw_root_dir,
        manifest_root_dir=manifest_root_dir,
        page_limit=page_limit,
        api_client_factory=sensor_api_client_factory,
    )
    hourly_ingestion = ingest_hourly_counts_task(
        raw_root_dir=raw_root_dir,
        manifest_root_dir=manifest_root_dir,
        date_range=date_range,
        api_client_factory=hourly_api_client_factory,
    )

    sensor_report = validate_snapshot_task(
        "sensor_locations",
        sensor_ingestion.snapshot_path,
        report_root_dir=report_root_dir,
    )
    hourly_report = validate_snapshot_task(
        "hourly_counts",
        hourly_ingestion.snapshot_path,
        report_root_dir=report_root_dir,
    )

    database_loads: tuple[DatabaseFlowResult, ...] = ()
    if load_to_database and database_url is not None:
        database_loads = load_snapshots_to_database_task(
            database_url=database_url,
            sensor_snapshot_path=sensor_ingestion.snapshot_path,
            hourly_snapshot_path=hourly_ingestion.snapshot_path,
        )

    return IngestionFlowResult(
        sensor_locations=_snapshot_result(
            dataset="sensor_locations",
            snapshot_path=sensor_ingestion.snapshot_path,
            manifest_path=sensor_ingestion.manifest_path,
            record_count=sensor_ingestion.record_count,
            report=sensor_report,
        ),
        hourly_counts=_snapshot_result(
            dataset="hourly_counts",
            snapshot_path=hourly_ingestion.snapshot_path,
            manifest_path=hourly_ingestion.manifest_path,
            record_count=hourly_ingestion.record_count,
            report=hourly_report,
        ),
        database_loads=database_loads,
    )
