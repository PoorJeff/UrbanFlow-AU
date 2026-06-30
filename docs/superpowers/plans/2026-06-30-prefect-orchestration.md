# Prefect Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local Prefect 3 ingestion flow that orchestrates existing UrbanFlow AU ingestion, validation, and optional PostgreSQL loading.

**Architecture:** Keep Prefect at the workflow boundary. The new `urbanflow.orchestration` package will wrap the existing ingestion, validation, and database modules in thin `@task` functions and expose one `@flow` entry point plus a testable CLI wrapper. Automated tests remain local and deterministic by monkeypatching task dependencies instead of calling the Melbourne API, PostgreSQL, or a Prefect server.

**Tech Stack:** Python 3.11+, Prefect 3, httpx, SQLAlchemy, pytest, Ruff.

---

## Source spec

Implement the approved design in
`docs/superpowers/specs/2026-06-30-prefect-orchestration-design.md`.

Official Prefect references used by this plan:

- Prefect flows: https://docs.prefect.io/v3/concepts/flows
- Prefect tasks: https://docs.prefect.io/v3/concepts/tasks
- Prefect deployments: https://docs.prefect.io/v3/deploy

## File structure

- Modify `pyproject.toml`
  - Add `prefect>=3,<4` to runtime dependencies.
- Create `src/urbanflow/orchestration/__init__.py`
  - Export orchestration result dataclasses and `run_ingestion_flow`.
- Create `src/urbanflow/orchestration/ingestion_flow.py`
  - Own result dataclasses, flow configuration errors, Prefect tasks, and the public flow.
- Create `src/urbanflow/orchestration/cli.py`
  - Own CLI parsing, JSON summary output, exit codes, and database URL resolution.
- Create `scripts/run_ingestion_flow.py`
  - One-line executable wrapper around `urbanflow.orchestration.cli.main`.
- Create `tests/unit/orchestration/test_ingestion_flow.py`
  - Cover date-range validation, no-database orchestration, validation failure behavior, and database loading order with monkeypatched task functions.
- Create `tests/unit/orchestration/test_ingestion_flow_cli.py`
  - Cover CLI invalid input, happy-path JSON output, database URL requirements, and script help.
- Modify `README.md`
  - Document the local Prefect ingestion flow command and optional database loading.

## Task 1: Add Prefect dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the runtime dependency**

Edit the `[project].dependencies` list in `pyproject.toml` so it includes Prefect:

```toml
dependencies = [
    "alembic>=1.13,<2",
    "httpx>=0.28,<1",
    "pandas>=2.1,<4",
    "pandera[pandas]>=0.24,<1",
    "prefect>=3,<4",
    "psycopg[binary]>=3.2,<4",
    "SQLAlchemy>=2.0,<3",
    "tenacity>=9,<10",
]
```

- [ ] **Step 2: Reinstall editable dependencies**

Run:

```powershell
python -m pip install -e ".[dev]"
```

Expected: install succeeds and `prefect` is available in the active virtual environment.

- [ ] **Step 3: Verify the dependency import**

Run:

```powershell
python -c "from prefect import flow, task; print(flow, task)"
```

Expected: command exits `0` and prints two callable objects.

- [ ] **Step 4: Commit dependency change**

Run:

```powershell
git add pyproject.toml
git commit -m "build: add Prefect dependency"
```

Expected: one commit containing only `pyproject.toml`.

## Task 2: Core flow result contract and no-database orchestration

**Files:**
- Create: `src/urbanflow/orchestration/__init__.py`
- Create: `src/urbanflow/orchestration/ingestion_flow.py`
- Create: `tests/unit/orchestration/test_ingestion_flow.py`

- [ ] **Step 1: Write failing flow tests**

Create `tests/unit/orchestration/test_ingestion_flow.py`:

```python
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

    result = ingestion_flow.run_ingestion_flow(
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
        ingestion_flow.run_ingestion_flow(
            raw_root_dir=tmp_path / "raw",
            manifest_root_dir=tmp_path / "manifests",
            report_root_dir=tmp_path / "reports",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
        )
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/orchestration/test_ingestion_flow.py -v
```

Expected: FAIL during collection with `ModuleNotFoundError` or `ImportError` for
`urbanflow.orchestration`.

- [ ] **Step 3: Implement the orchestration package exports**

Create `src/urbanflow/orchestration/__init__.py`:

```python
from urbanflow.orchestration.ingestion_flow import (
    DatabaseFlowResult,
    IngestionFlowConfigError,
    IngestionFlowError,
    IngestionFlowResult,
    SnapshotFlowResult,
    date_range_from_options,
    run_ingestion_flow,
)

__all__ = [
    "DatabaseFlowResult",
    "IngestionFlowConfigError",
    "IngestionFlowError",
    "IngestionFlowResult",
    "SnapshotFlowResult",
    "date_range_from_options",
    "run_ingestion_flow",
]
```

- [ ] **Step 4: Implement the core flow module**

Create `src/urbanflow/orchestration/ingestion_flow.py`:

```python
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
        raise IngestionFlowConfigError(
            "provide either --year or --start-date/--end-date, not both"
        )
    if has_year:
        return year_date_range(year)
    if has_start != has_end:
        raise IngestionFlowConfigError("provide both --start-date and --end-date")
    if not has_start or start_date is None or end_date is None:
        raise IngestionFlowConfigError("provide --year or both --start-date and --end-date")
    return validate_date_range(start_date, end_date)


def _snapshot_result(
    *,
    dataset: str,
    snapshot_path: Path,
    manifest_path: Path,
    record_count: int,
    report: ValidationReport,
) -> SnapshotFlowResult:
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
    api_client_factory: SensorApiClientFactory = _default_sensor_api_client_factory,
) -> SensorLocationIngestionResult:
    with httpx.Client(timeout=30.0) as http_client:
        return ingest_sensor_locations(
            api_client=api_client_factory(http_client),
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
    api_client_factory: HourlyApiClientFactory = _default_hourly_api_client_factory,
) -> HourlyCountIngestionResult:
    with httpx.Client(timeout=30.0) as http_client:
        return ingest_hourly_counts(
            api_client=api_client_factory(http_client),
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
    if not report.passed:
        codes = ", ".join(issue.code for issue in report.errors)
        raise IngestionFlowError(f"Validation failed for {dataset}: {codes}")
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
    sensor_api_client_factory: SensorApiClientFactory = _default_sensor_api_client_factory,
    hourly_api_client_factory: HourlyApiClientFactory = _default_hourly_api_client_factory,
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
```

- [ ] **Step 5: Run flow tests to verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/orchestration/test_ingestion_flow.py -v
```

Expected: all tests in `test_ingestion_flow.py` pass.

- [ ] **Step 6: Run focused lint and format checks**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check src/urbanflow/orchestration tests/unit/orchestration --no-cache
python -m ruff format --check src/urbanflow/orchestration tests/unit/orchestration
```

Expected: no Ruff diagnostics and no formatting changes required.

- [ ] **Step 7: Commit core flow**

Run:

```powershell
git add src/urbanflow/orchestration tests/unit/orchestration/test_ingestion_flow.py
git commit -m "feat: add local Prefect ingestion flow"
```

Expected: one commit containing the new orchestration package and flow tests.

## Task 3: Database loading behavior

**Files:**
- Modify: `tests/unit/orchestration/test_ingestion_flow.py`
- Modify: `src/urbanflow/orchestration/ingestion_flow.py`

- [ ] **Step 1: Add a failing database loading unit test**

Append this test to `tests/unit/orchestration/test_ingestion_flow.py`:

```python
from urbanflow.database.loaders import DatabaseLoadResult


def test_load_snapshots_to_database_loads_sensor_before_hourly(monkeypatch, tmp_path):
    calls = []
    sensor_snapshot_path = tmp_path / "sensor" / "records.json"
    hourly_snapshot_path = tmp_path / "hourly" / "records.csv"

    class FakeEngine:
        def dispose(self):
            calls.append(("dispose",))

    class FakeSessionFactory:
        def begin(self):
            class Context:
                def __enter__(self):
                    calls.append(("begin",))
                    return "session"

                def __exit__(self, exc_type, exc, tb):
                    calls.append(("end",))
                    return False

            return Context()

    def fake_create_database_engine(database_url):
        calls.append(("engine", database_url))
        return FakeEngine()

    def fake_create_session_factory(engine):
        calls.append(("factory", type(engine).__name__))
        return FakeSessionFactory()

    def fake_load_sensor_locations_snapshot(session, snapshot_path):
        calls.append(("sensor_load", session, snapshot_path))
        return DatabaseLoadResult(
            dataset="sensor_locations",
            row_count=2,
            validation_warning_count=0,
        )

    def fake_load_hourly_counts_snapshot(session, snapshot_path):
        calls.append(("hourly_load", session, snapshot_path))
        return DatabaseLoadResult(
            dataset="hourly_counts",
            row_count=3,
            validation_warning_count=1,
        )

    monkeypatch.setattr(
        ingestion_flow,
        "create_database_engine",
        fake_create_database_engine,
    )
    monkeypatch.setattr(
        ingestion_flow,
        "create_session_factory",
        fake_create_session_factory,
    )
    monkeypatch.setattr(
        ingestion_flow,
        "load_sensor_locations_snapshot",
        fake_load_sensor_locations_snapshot,
    )
    monkeypatch.setattr(
        ingestion_flow,
        "load_hourly_counts_snapshot",
        fake_load_hourly_counts_snapshot,
    )

    result = ingestion_flow.load_snapshots_to_database(
        database_url="postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
        sensor_snapshot_path=sensor_snapshot_path,
        hourly_snapshot_path=hourly_snapshot_path,
    )

    assert result == (
        ingestion_flow.DatabaseFlowResult(
            dataset="sensor_locations",
            row_count=2,
            validation_warning_count=0,
        ),
        ingestion_flow.DatabaseFlowResult(
            dataset="hourly_counts",
            row_count=3,
            validation_warning_count=1,
        ),
    )
    assert calls == [
        (
            "engine",
            "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
        ),
        ("factory", "FakeEngine"),
        ("begin",),
        ("sensor_load", "session", sensor_snapshot_path),
        ("hourly_load", "session", hourly_snapshot_path),
        ("end",),
        ("dispose",),
    ]


def test_run_ingestion_flow_loads_database_after_validation(monkeypatch, tmp_path):
    calls = []

    def fake_ingest_sensor_locations_task(**kwargs):
        calls.append(("sensor_ingest",))
        return sensor_result(tmp_path)

    def fake_ingest_hourly_counts_task(**kwargs):
        calls.append(("hourly_ingest",))
        return hourly_result(tmp_path, kwargs["date_range"])

    def fake_validate_snapshot_task(dataset, snapshot_path, *, report_root_dir):
        calls.append(("validate", dataset))
        return validation_report(dataset, snapshot_path)

    def fake_load_snapshots_to_database_task(**kwargs):
        calls.append(
            (
                "database_load",
                kwargs["database_url"],
                kwargs["sensor_snapshot_path"],
                kwargs["hourly_snapshot_path"],
            )
        )
        return (
            ingestion_flow.DatabaseFlowResult(
                dataset="sensor_locations",
                row_count=2,
                validation_warning_count=0,
            ),
            ingestion_flow.DatabaseFlowResult(
                dataset="hourly_counts",
                row_count=3,
                validation_warning_count=0,
            ),
        )

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
        fake_load_snapshots_to_database_task,
    )

    result = ingestion_flow.run_ingestion_flow(
        raw_root_dir=tmp_path / "raw",
        manifest_root_dir=tmp_path / "manifests",
        report_root_dir=tmp_path / "reports",
        year=2025,
        load_to_database=True,
        database_url="postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
    )

    assert result.database_loads == (
        ingestion_flow.DatabaseFlowResult(
            dataset="sensor_locations",
            row_count=2,
            validation_warning_count=0,
        ),
        ingestion_flow.DatabaseFlowResult(
            dataset="hourly_counts",
            row_count=3,
            validation_warning_count=0,
        ),
    )
    assert calls == [
        ("sensor_ingest",),
        ("hourly_ingest",),
        ("validate", "sensor_locations"),
        ("validate", "hourly_counts"),
        (
            "database_load",
            "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow",
            tmp_path / "raw" / "sensor_locations" / "records.json",
            tmp_path / "raw" / "hourly_counts" / "records.csv",
        ),
    ]
```

- [ ] **Step 2: Run database tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/orchestration/test_ingestion_flow.py::test_load_snapshots_to_database_loads_sensor_before_hourly tests/unit/orchestration/test_ingestion_flow.py::test_run_ingestion_flow_loads_database_after_validation -v
```

Expected: PASS if Task 2 implementation already included the database helper.
If a test fails, fix only `src/urbanflow/orchestration/ingestion_flow.py` to match the
behavior in the test.

- [ ] **Step 3: Run all orchestration flow tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/orchestration/test_ingestion_flow.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Run focused checks**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check src/urbanflow/orchestration tests/unit/orchestration --no-cache
python -m ruff format --check src/urbanflow/orchestration tests/unit/orchestration
```

Expected: no Ruff diagnostics and no formatting changes required.

- [ ] **Step 5: Commit database orchestration coverage**

If Task 3 changed files, run:

```powershell
git add src/urbanflow/orchestration/ingestion_flow.py tests/unit/orchestration/test_ingestion_flow.py
git commit -m "test: cover Prefect database loading order"
```

Expected: one commit with the database loading test and any required implementation adjustment.

If Task 3 made no tracked changes because Task 2 already satisfied these tests, do not create an empty commit.

## Task 4: CLI wrapper and script

**Files:**
- Create: `src/urbanflow/orchestration/cli.py`
- Create: `scripts/run_ingestion_flow.py`
- Create: `tests/unit/orchestration/test_ingestion_flow_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/unit/orchestration/test_ingestion_flow_cli.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from urbanflow.orchestration import cli
from urbanflow.orchestration.ingestion_flow import (
    DatabaseFlowResult,
    IngestionFlowResult,
    SnapshotFlowResult,
)


def flow_result() -> IngestionFlowResult:
    return IngestionFlowResult(
        sensor_locations=SnapshotFlowResult(
            dataset="sensor_locations",
            snapshot_path="data/raw/sensor_locations/records.json",
            manifest_path="data/manifests/sensor_locations.json",
            record_count=2,
            validation_passed=True,
            validation_error_count=0,
            validation_warning_count=0,
        ),
        hourly_counts=SnapshotFlowResult(
            dataset="hourly_counts",
            snapshot_path="data/raw/hourly_counts/records.csv",
            manifest_path="data/manifests/hourly_counts.json",
            record_count=3,
            validation_passed=True,
            validation_error_count=0,
            validation_warning_count=1,
        ),
        database_loads=(
            DatabaseFlowResult(
                dataset="sensor_locations",
                row_count=2,
                validation_warning_count=0,
            ),
        ),
    )


def test_ingestion_flow_cli_returns_two_when_date_range_missing(capsys):
    exit_code = cli.main([], environ={})

    assert exit_code == 2
    assert "provide --year" in capsys.readouterr().err


def test_ingestion_flow_cli_runs_flow_without_database(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_run_ingestion_flow(**kwargs):
        calls.update(kwargs)
        return flow_result()

    monkeypatch.setattr(cli, "run_ingestion_flow", fake_run_ingestion_flow)

    exit_code = cli.main(
        [
            "--raw-root",
            str(tmp_path / "raw"),
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--report-root",
            str(tmp_path / "reports"),
            "--year",
            "2025",
            "--page-limit",
            "25",
        ],
        environ={},
    )

    assert exit_code == 0
    assert calls["raw_root_dir"] == tmp_path / "raw"
    assert calls["manifest_root_dir"] == tmp_path / "manifests"
    assert calls["report_root_dir"] == tmp_path / "reports"
    assert calls["year"] == 2025
    assert calls["start_date"] is None
    assert calls["end_date"] is None
    assert calls["page_limit"] == 25
    assert calls["load_to_database"] is False
    assert calls["database_url"] is None
    assert json.loads(capsys.readouterr().out)["hourly_counts"]["record_count"] == 3


def test_ingestion_flow_cli_requires_database_url_when_loading(capsys):
    exit_code = cli.main(["--year", "2025", "--load-to-database"], environ={})

    assert exit_code == 2
    assert "Database URL is required" in capsys.readouterr().err


def test_ingestion_flow_cli_uses_database_url_from_environment(monkeypatch, capsys):
    calls = {}

    def fake_run_ingestion_flow(**kwargs):
        calls.update(kwargs)
        return flow_result()

    monkeypatch.setattr(cli, "run_ingestion_flow", fake_run_ingestion_flow)

    exit_code = cli.main(
        ["--year", "2025", "--load-to-database"],
        environ={"URBANFLOW_DATABASE_URL": "postgresql+psycopg://env"},
    )

    assert exit_code == 0
    assert calls["load_to_database"] is True
    assert calls["database_url"] == "postgresql+psycopg://env"
    assert json.loads(capsys.readouterr().out)["database_loads"][0]["row_count"] == 2


def test_ingestion_flow_script_help() -> None:
    import subprocess
    import sys

    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, repository_root / "scripts" / "run_ingestion_flow.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run the UrbanFlow AU Prefect ingestion flow" in result.stdout
```

- [ ] **Step 2: Run CLI tests to verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/orchestration/test_ingestion_flow_cli.py -v
```

Expected: FAIL because `urbanflow.orchestration.cli` and
`scripts/run_ingestion_flow.py` do not exist.

- [ ] **Step 3: Implement the CLI module**

Create `src/urbanflow/orchestration/cli.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path

from urbanflow.database.config import DatabaseConfigError, get_database_url
from urbanflow.ingestion.hourly_counts import (
    HourlyCountIngestionError,
    parse_iso_date,
)
from urbanflow.ingestion.melbourne_api import MelbourneApiError
from urbanflow.ingestion.sensor_locations import SensorLocationParseError
from urbanflow.orchestration.ingestion_flow import (
    IngestionFlowConfigError,
    IngestionFlowError,
    IngestionFlowResult,
    run_ingestion_flow,
)
from urbanflow.validation.pipeline import ValidationPipelineError


def positive_integer(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed_value


def parse_optional_date(value: str | None) -> date | None:
    if value is None:
        return None
    return parse_iso_date(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the UrbanFlow AU Prefect ingestion flow.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/data_quality"))
    parser.add_argument("--year", type=int)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--page-limit", type=positive_integer, default=100)
    parser.add_argument("--load-to-database", action="store_true")
    parser.add_argument("--database-url", default=None)
    return parser


def result_summary(result: IngestionFlowResult) -> dict[str, object]:
    return asdict(result)


def _resolve_database_url(
    *,
    load_to_database: bool,
    database_url: str | None,
    environ: Mapping[str, str] | None,
) -> str | None:
    if not load_to_database:
        return None
    return get_database_url(database_url, environ=environ)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        start_date = parse_optional_date(args.start_date)
        end_date = parse_optional_date(args.end_date)
        resolved_database_url = _resolve_database_url(
            load_to_database=args.load_to_database,
            database_url=args.database_url,
            environ=environ,
        )
        result = run_ingestion_flow(
            raw_root_dir=args.raw_root,
            manifest_root_dir=args.manifest_root,
            report_root_dir=args.report_root,
            year=args.year,
            start_date=start_date,
            end_date=end_date,
            page_limit=args.page_limit,
            load_to_database=args.load_to_database,
            database_url=resolved_database_url,
        )
    except (
        DatabaseConfigError,
        HourlyCountIngestionError,
        IngestionFlowConfigError,
        argparse.ArgumentTypeError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (
        IngestionFlowError,
        MelbourneApiError,
        OSError,
        SensorLocationParseError,
        ValidationPipelineError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result_summary(result), sort_keys=True))
    return 0
```

- [ ] **Step 4: Add the script wrapper**

Create `scripts/run_ingestion_flow.py`:

```python
from urbanflow.orchestration.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run CLI tests to verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/orchestration/test_ingestion_flow_cli.py -v
```

Expected: all CLI tests pass.

- [ ] **Step 6: Run all orchestration tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/orchestration -v
```

Expected: all orchestration tests pass.

- [ ] **Step 7: Commit CLI implementation**

Run:

```powershell
git add src/urbanflow/orchestration/cli.py scripts/run_ingestion_flow.py tests/unit/orchestration/test_ingestion_flow_cli.py
git commit -m "feat: add Prefect ingestion flow CLI"
```

Expected: one commit containing the CLI, script wrapper, and CLI tests.

## Task 5: README documentation and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with local flow command**

Add this section after the existing "Run hourly-count ingestion locally" section and before
"Validate a local raw snapshot":

````markdown
## Run the local Prefect ingestion flow

```powershell
python scripts/run_ingestion_flow.py --year 2025
```

The command runs the local Prefect flow for sensor-location ingestion, bounded
hourly-count ingestion, and snapshot validation. It writes raw snapshots below
`data/raw/`, manifests below `data/manifests/`, and validation reports below
`reports/data_quality/`.

To also load the generated snapshots into PostgreSQL, run migrations first and
pass a database URL explicitly or through `URBANFLOW_DATABASE_URL`:

```powershell
$env:URBANFLOW_DATABASE_URL = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"
alembic upgrade head
python scripts/run_ingestion_flow.py --year 2025 --load-to-database
```

The flow is local by design. It does not require a Prefect server, deployment,
work pool, or schedule.
````

- [ ] **Step 2: Run README script help smoke**

Run:

```powershell
$env:PYTHONPATH='src'
python scripts/run_ingestion_flow.py --help
```

Expected: command exits `0` and prints `Run the UrbanFlow AU Prefect ingestion flow`.

- [ ] **Step 3: Run full verification**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check . --no-cache
python -m ruff format --check .
python -m pytest
```

Expected: Ruff passes and all tests pass.

- [ ] **Step 4: Confirm no accidental live service requirement**

Run:

```powershell
if ($env:URBANFLOW_DATABASE_URL) { "URBANFLOW_DATABASE_URL is set" } else { "URBANFLOW_DATABASE_URL is not set" }
```

Expected: the verification does not depend on this value. If it is not set, automated tests still pass.

- [ ] **Step 5: Commit documentation**

Run:

```powershell
git add README.md
git commit -m "docs: document local Prefect ingestion flow"
```

Expected: one documentation commit.

## Task 6: Merge, push, and cleanup

**Files:**
- No new source files.

- [ ] **Step 1: Verify branch status**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree on the local implementation branch.

- [ ] **Step 2: Merge to main only**

From repository root `D:\Github项目\UrbanFlow-AU`, run:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 fetch origin main
git merge --ff-only codex/prefect-orchestration
```

Expected: `main` fast-forwards to the implementation branch.

- [ ] **Step 3: Re-run final checks on main**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check . --no-cache
python -m ruff format --check .
python -m pytest
```

Expected: Ruff passes and all tests pass on `main`.

- [ ] **Step 4: Push main**

Run:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 push origin main
```

Expected: only `main` is pushed to GitHub.

- [ ] **Step 5: Remove the local worktree and local codex branch**

Before removal, verify the resolved path stays under
`D:\Github项目\UrbanFlow-AU\.worktrees`:

```powershell
$target = (Resolve-Path '.worktrees\prefect-orchestration').Path
$root = (Resolve-Path '.worktrees').Path
if (-not $target.StartsWith($root)) { throw "Refusing to remove worktree outside .worktrees: $target" }
git worktree remove $target
git worktree prune
git branch -d codex/prefect-orchestration
```

Expected: local feature worktree and local codex branch are removed after the successful merge and push.

## Self-review checklist

- Spec coverage:
  - Prefect dependency: Task 1.
  - Local flow over existing ingestion, validation, and optional database loading: Tasks 2 and 3.
  - CLI script and README command: Tasks 4 and 5.
  - No network, PostgreSQL, Prefect server, deployment, schedule, or work pool requirement in automated tests: Tasks 2, 3, 4, and 5.
- Placeholder scan:
  - The plan contains no unresolved markers.
  - Each code-changing step includes concrete code.
- Type consistency:
  - `IngestionFlowResult`, `SnapshotFlowResult`, `DatabaseFlowResult`,
    `IngestionFlowConfigError`, and `IngestionFlowError` are defined before use.
  - CLI tests and flow tests use the same dataclass field names as the implementation.
