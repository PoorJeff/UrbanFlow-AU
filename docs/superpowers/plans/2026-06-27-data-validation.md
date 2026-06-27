# Data Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Pandera-backed validation layer for local `sensor_locations` JSON snapshots and `hourly_counts` CSV snapshots.

**Architecture:** Create a focused `urbanflow.validation` package with stable report dataclasses, snapshot readers, dataset validators, a routing pipeline, and a CLI wrapper. Keep raw ingestion independent from pandas/Pandera; validation reads completed snapshots and writes optional JSON quality reports.

**Tech Stack:** Python 3.11+, pandas, pandera with the pandas extra, pytest, Ruff.

---

## File Structure

- Modify `pyproject.toml`
  - Add runtime dependencies for `pandas` and `pandera[pandas]`.
- Create `src/urbanflow/validation/__init__.py`
  - Export the public validation API.
- Create `src/urbanflow/validation/reports.py`
  - Define stable report dataclasses and JSON report writing.
- Create `src/urbanflow/validation/snapshot_readers.py`
  - Load local JSON/CSV snapshots into pandas DataFrames and raise read errors.
- Create `src/urbanflow/validation/sensor_locations.py`
  - Validate sensor-location DataFrames and snapshots.
- Create `src/urbanflow/validation/hourly_counts.py`
  - Validate hourly-count DataFrames and snapshots.
- Create `src/urbanflow/validation/pipeline.py`
  - Route dataset names to validators and optionally write reports.
- Create `src/urbanflow/validation/cli.py`
  - Provide a testable CLI entry point.
- Create `scripts/validate_snapshot.py`
  - Thin script wrapper for the CLI module.
- Modify `README.md`
  - Document the local validation command.
- Create tests under `tests/unit/validation/`
  - Cover reports, readers, dataset validators, pipeline routing, and CLI behavior.

## Task 1: Report Contract

**Files:**
- Create: `tests/unit/validation/test_reports.py`
- Create: `src/urbanflow/validation/__init__.py`
- Create: `src/urbanflow/validation/reports.py`

- [ ] **Step 1: Write the failing report tests**

Create `tests/unit/validation/test_reports.py`:

```python
from datetime import UTC, datetime
import json

import pytest

from urbanflow.validation.reports import (
    ValidationIssue,
    ValidationMetric,
    ValidationReport,
)


def test_validation_report_serializes_stable_shape(tmp_path):
    report = ValidationReport(
        dataset="sensor_locations",
        snapshot_path="data/raw/melbourne/sensor_locations/example/records.json",
        validated_at=datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
        row_count=2,
        errors=(
            ValidationIssue(
                code="DUPLICATE_LOCATION_ID",
                message="location_id values must be unique",
                column="location_id",
                rows=(1,),
            ),
        ),
        warnings=(
            ValidationIssue(
                code="NULL_INSTALLATION_DATE",
                message="installation_date is null for 1 row",
                column="installation_date",
            ),
        ),
        metrics=(
            ValidationMetric(name="sensor_count", value=2),
            ValidationMetric(name="status_distribution", value={"A": 2}),
        ),
    )

    payload = report.to_dict()

    assert payload == {
        "schema_version": 1,
        "dataset": "sensor_locations",
        "snapshot_path": "data/raw/melbourne/sensor_locations/example/records.json",
        "validated_at": "2026-06-27T12:00:00Z",
        "passed": False,
        "row_count": 2,
        "errors": [
            {
                "code": "DUPLICATE_LOCATION_ID",
                "message": "location_id values must be unique",
                "column": "location_id",
                "rows": [1],
            }
        ],
        "warnings": [
            {
                "code": "NULL_INSTALLATION_DATE",
                "message": "installation_date is null for 1 row",
                "column": "installation_date",
                "rows": [],
            }
        ],
        "metrics": {
            "sensor_count": 2,
            "status_distribution": {"A": 2},
        },
    }

    output_path = tmp_path / "quality" / "sensor_locations" / "report.json"
    report.write_json(output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_validation_report_refuses_to_overwrite(tmp_path):
    report = ValidationReport(
        dataset="hourly_counts",
        snapshot_path="records.csv",
        validated_at=datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
        row_count=0,
    )
    output_path = tmp_path / "report.json"
    output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Validation report already exists"):
        report.write_json(output_path)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/validation/test_reports.py -v
```

Expected: collection fails with `ModuleNotFoundError` for `urbanflow.validation`.

- [ ] **Step 3: Implement report dataclasses**

Create `src/urbanflow/validation/reports.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_utc_timestamp(value: datetime) -> str:
    timestamp = value.astimezone(UTC)
    return timestamp.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    column: str | None = None
    rows: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "code": self.code,
            "message": self.message,
            "column": self.column,
            "rows": list(self.rows),
        }
        return payload


@dataclass(frozen=True)
class ValidationMetric:
    name: str
    value: JsonValue


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    snapshot_path: str
    validated_at: datetime
    row_count: int
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    metrics: tuple[ValidationMetric, ...] = ()
    schema_version: int = field(default=1, init=False)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "snapshot_path": self.snapshot_path,
            "validated_at": format_utc_timestamp(self.validated_at),
            "passed": self.passed,
            "row_count": self.row_count,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "metrics": {metric.name: metric.value for metric in self.metrics},
        }

    def write_json(self, output_path: Path) -> Path:
        if output_path.exists():
            raise FileExistsError(f"Validation report already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path
```

Create `src/urbanflow/validation/__init__.py`:

```python
"""Data validation utilities for UrbanFlow AU snapshots."""

from urbanflow.validation.reports import ValidationIssue, ValidationMetric, ValidationReport

__all__ = ["ValidationIssue", "ValidationMetric", "ValidationReport"]
```

- [ ] **Step 4: Run focused tests and commit**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/validation/test_reports.py -v
```

Expected: both report tests pass.

Commit:

```powershell
git add src/urbanflow/validation/__init__.py src/urbanflow/validation/reports.py tests/unit/validation/test_reports.py
git commit -m "feat: add validation report contract"
```

## Task 2: Snapshot Readers and Pipeline Routing

**Files:**
- Create: `tests/unit/validation/test_snapshot_readers.py`
- Create: `tests/unit/validation/test_pipeline.py`
- Create: `src/urbanflow/validation/snapshot_readers.py`
- Create: `src/urbanflow/validation/pipeline.py`
- Modify: `src/urbanflow/validation/__init__.py`

- [ ] **Step 1: Write failing reader and pipeline tests**

Create `tests/unit/validation/test_snapshot_readers.py`:

```python
import json

import pytest

from urbanflow.validation.snapshot_readers import (
    SnapshotReadError,
    read_hourly_counts_snapshot,
    read_sensor_locations_snapshot,
)


def test_read_sensor_locations_snapshot_loads_json_records(tmp_path):
    snapshot_path = tmp_path / "records.json"
    snapshot_path.write_text(
        json.dumps(
            [
                {
                    "location_id": 1,
                    "sensor_description": "Bourke Street",
                    "sensor_name": "Sensor A",
                    "installation_date": None,
                    "status": "A",
                    "latitude": -37.81,
                    "longitude": 144.96,
                }
            ]
        ),
        encoding="utf-8",
    )

    frame = read_sensor_locations_snapshot(snapshot_path)

    assert frame.to_dict("records")[0]["sensor_name"] == "Sensor A"


def test_read_sensor_locations_snapshot_rejects_non_list_json(tmp_path):
    snapshot_path = tmp_path / "records.json"
    snapshot_path.write_text(json.dumps({"records": []}), encoding="utf-8")

    with pytest.raises(SnapshotReadError, match="JSON snapshot must contain a list"):
        read_sensor_locations_snapshot(snapshot_path)


def test_read_hourly_counts_snapshot_loads_csv_as_strings(tmp_path):
    snapshot_path = tmp_path / "records.csv"
    snapshot_path.write_text(
        "id,location_id,sensing_date,hourday,direction_1,direction_2,pedestriancount,sensor_name,location\n"
        "abc,1,2025-01-01,0,2,3,5,Sensor A,-37.81,144.96\n",
        encoding="utf-8",
    )

    frame = read_hourly_counts_snapshot(snapshot_path)

    assert frame.loc[0, "hourday"] == "0"
    assert frame.loc[0, "pedestriancount"] == "5"


def test_snapshot_reader_reports_missing_file(tmp_path):
    with pytest.raises(SnapshotReadError, match="Snapshot file does not exist"):
        read_hourly_counts_snapshot(tmp_path / "missing.csv")
```

Create `tests/unit/validation/test_pipeline.py`:

```python
from datetime import UTC, datetime
import json

import pytest

from urbanflow.validation.pipeline import ValidationPipelineError, validate_snapshot


def test_validate_snapshot_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValidationPipelineError, match="Unsupported dataset"):
        validate_snapshot("unknown", tmp_path / "records.json")


def test_validate_snapshot_returns_read_error_report_for_unreadable_snapshot(tmp_path):
    report = validate_snapshot(
        "sensor_locations",
        tmp_path / "missing.json",
        validated_at=datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
    )

    assert report.passed is False
    assert report.row_count == 0
    assert report.errors[0].code == "SNAPSHOT_READ_ERROR"


def test_validate_snapshot_writes_report_when_report_root_is_provided(tmp_path):
    snapshot_path = tmp_path / "records.json"
    snapshot_path.write_text(
        json.dumps(
            [
                {
                    "location_id": 1,
                    "sensor_description": "Bourke Street",
                    "sensor_name": "Sensor A",
                    "installation_date": None,
                    "status": "A",
                    "latitude": -37.81,
                    "longitude": 144.96,
                }
            ]
        ),
        encoding="utf-8",
    )

    report = validate_snapshot(
        "sensor_locations",
        snapshot_path,
        report_root=tmp_path / "reports",
        validated_at=datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
    )

    assert report.passed is True
    report_path = tmp_path / "reports" / "sensor_locations" / "20260627T120000Z.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
```

- [ ] **Step 2: Install pandas/Pandera in the local environment**

Run:

```powershell
& ..\..\.venv\Scripts\python.exe -m pip install "pandera[pandas]>=0.24,<1"
```

Expected: pip installs or reports the requirement already satisfied.

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/validation/test_snapshot_readers.py tests/unit/validation/test_pipeline.py -v
```

Expected: collection fails because `urbanflow.validation.snapshot_readers` and
`urbanflow.validation.pipeline` do not exist.

- [ ] **Step 4: Implement readers and pipeline**

Create `src/urbanflow/validation/snapshot_readers.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class SnapshotReadError(Exception):
    """Raised when a local snapshot cannot be loaded for validation."""


def _ensure_existing_file(snapshot_path: Path) -> None:
    if not snapshot_path.exists():
        raise SnapshotReadError(f"Snapshot file does not exist: {snapshot_path}")
    if not snapshot_path.is_file():
        raise SnapshotReadError(f"Snapshot path is not a file: {snapshot_path}")


def read_sensor_locations_snapshot(snapshot_path: Path) -> pd.DataFrame:
    _ensure_existing_file(snapshot_path)
    try:
        payload: Any = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotReadError(f"Could not parse JSON snapshot: {snapshot_path}") from exc
    if not isinstance(payload, list):
        raise SnapshotReadError("JSON snapshot must contain a list of records")
    return pd.DataFrame.from_records(payload)


def read_hourly_counts_snapshot(snapshot_path: Path) -> pd.DataFrame:
    _ensure_existing_file(snapshot_path)
    try:
        return pd.read_csv(snapshot_path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise SnapshotReadError(f"CSV snapshot is empty: {snapshot_path}") from exc
    except UnicodeDecodeError as exc:
        raise SnapshotReadError(f"Could not decode CSV snapshot: {snapshot_path}") from exc
```

Create `src/urbanflow/validation/pipeline.py`:

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from urbanflow.validation.reports import ValidationIssue, ValidationReport, utc_now
from urbanflow.validation.snapshot_readers import SnapshotReadError


class ValidationPipelineError(Exception):
    """Raised when validation cannot be routed or configured."""


SnapshotValidator = Callable[[Path, datetime | None], ValidationReport]


def _report_path(report_root: Path, dataset: str, validated_at: datetime) -> Path:
    timestamp = validated_at.strftime("%Y%m%dT%H%M%SZ")
    return report_root / dataset / f"{timestamp}.json"


def _read_error_report(dataset: str, snapshot_path: Path, exc: SnapshotReadError, validated_at: datetime) -> ValidationReport:
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


def validate_snapshot(
    dataset: str,
    snapshot_path: Path,
    *,
    report_root: Path | None = None,
    validated_at: datetime | None = None,
) -> ValidationReport:
    from urbanflow.validation.hourly_counts import validate_hourly_counts_snapshot
    from urbanflow.validation.sensor_locations import validate_sensor_locations_snapshot

    validators: dict[str, SnapshotValidator] = {
        "sensor_locations": validate_sensor_locations_snapshot,
        "hourly_counts": validate_hourly_counts_snapshot,
    }
    if dataset not in validators:
        raise ValidationPipelineError(f"Unsupported dataset: {dataset}")

    timestamp = validated_at or utc_now()
    try:
        report = validators[dataset](snapshot_path, timestamp)
    except SnapshotReadError as exc:
        report = _read_error_report(dataset, snapshot_path, exc, timestamp)

    if report_root is not None:
        report.write_json(_report_path(report_root, dataset, report.validated_at))
    return report
```

Update `src/urbanflow/validation/__init__.py`:

```python
"""Data validation utilities for UrbanFlow AU snapshots."""

from urbanflow.validation.pipeline import validate_snapshot
from urbanflow.validation.reports import ValidationIssue, ValidationMetric, ValidationReport

__all__ = ["ValidationIssue", "ValidationMetric", "ValidationReport", "validate_snapshot"]
```

- [ ] **Step 5: Run focused tests and commit**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/validation/test_reports.py tests/unit/validation/test_snapshot_readers.py tests/unit/validation/test_pipeline.py -v
```

Expected: pipeline tests still fail until Task 3 creates the sensor validator and Task 4 creates
the hourly validator; reader tests pass.

Commit readers after Task 3 and Task 4 make the pipeline tests pass.

## Task 3: Sensor-Location Validation

**Files:**
- Create: `tests/unit/validation/test_sensor_locations.py`
- Create: `src/urbanflow/validation/sensor_locations.py`

- [ ] **Step 1: Write failing sensor-location tests**

Create `tests/unit/validation/test_sensor_locations.py`:

```python
import json
from datetime import UTC, datetime

from urbanflow.validation.sensor_locations import validate_sensor_locations_snapshot


def _write_snapshot(tmp_path, records):
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _valid_record(**overrides):
    record = {
        "location_id": 1,
        "sensor_description": "Bourke Street",
        "sensor_name": "Sensor A",
        "installation_date": None,
        "status": "A",
        "latitude": -37.81,
        "longitude": 144.96,
    }
    record.update(overrides)
    return record


def test_sensor_location_snapshot_passes_and_records_metrics(tmp_path):
    snapshot_path = _write_snapshot(
        tmp_path,
        [
            _valid_record(location_id=1, status="A"),
            _valid_record(location_id=2, status="I", installation_date="2020-01-01"),
        ],
    )

    report = validate_sensor_locations_snapshot(
        snapshot_path,
        datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
    )

    assert report.passed is True
    assert report.row_count == 2
    assert report.to_dict()["metrics"]["sensor_count"] == 2
    assert report.to_dict()["metrics"]["null_installation_date_count"] == 1
    assert report.to_dict()["metrics"]["status_distribution"] == {"A": 1, "I": 1}


def test_sensor_location_snapshot_fails_for_duplicate_location_id(tmp_path):
    snapshot_path = _write_snapshot(tmp_path, [_valid_record(), _valid_record()])

    report = validate_sensor_locations_snapshot(snapshot_path)

    assert report.passed is False
    assert any(issue.code == "DUPLICATE_LOCATION_ID" for issue in report.errors)


def test_sensor_location_snapshot_fails_for_blank_name_and_bad_coordinates(tmp_path):
    snapshot_path = _write_snapshot(
        tmp_path,
        [_valid_record(sensor_name=" ", latitude=-100, longitude=200)],
    )

    report = validate_sensor_locations_snapshot(snapshot_path)

    assert report.passed is False
    assert any(issue.code == "SCHEMA_INVALID" for issue in report.errors)
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/validation/test_sensor_locations.py -v
```

Expected: collection fails because `urbanflow.validation.sensor_locations` does not exist.

- [ ] **Step 3: Implement sensor-location validation**

Create `src/urbanflow/validation/sensor_locations.py`:

```python
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
        "installation_date": pa.Column(object, nullable=True),
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
                message=f"Sensor-location schema validation failed: {len(exc.failure_cases)} failure cases",
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
```

- [ ] **Step 4: Run sensor tests and partial pipeline tests**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/validation/test_sensor_locations.py tests/unit/validation/test_pipeline.py::test_validate_snapshot_writes_report_when_report_root_is_provided -v
```

Expected: sensor tests pass and the sensor pipeline report-writing test passes.

## Task 4: Hourly-Count Validation

**Files:**
- Create: `tests/unit/validation/test_hourly_counts.py`
- Create: `src/urbanflow/validation/hourly_counts.py`

- [ ] **Step 1: Write failing hourly-count tests**

Create `tests/unit/validation/test_hourly_counts.py`:

```python
from datetime import UTC, datetime

from urbanflow.validation.hourly_counts import validate_hourly_counts_snapshot


HEADER = "id,location_id,sensing_date,hourday,direction_1,direction_2,pedestriancount,sensor_name,location\n"


def _write_csv(tmp_path, rows):
    path = tmp_path / "records.csv"
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def test_hourly_count_snapshot_passes_and_records_metrics(tmp_path):
    snapshot_path = _write_csv(
        tmp_path,
        [
            "a,1,2025-01-01,0,2,3,5,Sensor A,-37.81,144.96\n",
            "b,1,2025-01-01,1,1,1,2,Sensor A,-37.81,144.96\n",
            "c,2,2025-01-02,23,4,6,10,Sensor B,-37.82,144.97\n",
        ],
    )

    report = validate_hourly_counts_snapshot(
        snapshot_path,
        datetime(2026, 6, 27, 12, 0, tzinfo=UTC),
    )

    assert report.passed is True
    payload = report.to_dict()
    assert payload["metrics"]["row_count"] == 3
    assert payload["metrics"]["sensor_count"] == 2
    assert payload["metrics"]["date_range"] == {"start": "2025-01-01", "end": "2025-01-02"}
    assert payload["metrics"]["hour_distribution"]["0"] == 1


def test_hourly_count_snapshot_fails_for_duplicate_id(tmp_path):
    snapshot_path = _write_csv(
        tmp_path,
        [
            "a,1,2025-01-01,0,2,3,5,Sensor A,-37.81,144.96\n",
            "a,1,2025-01-01,1,1,1,2,Sensor A,-37.81,144.96\n",
        ],
    )

    report = validate_hourly_counts_snapshot(snapshot_path)

    assert report.passed is False
    assert any(issue.code == "DUPLICATE_SOURCE_ID" for issue in report.errors)


def test_hourly_count_snapshot_fails_for_hour_range_and_direction_total(tmp_path):
    snapshot_path = _write_csv(
        tmp_path,
        ["a,1,2025-01-01,24,2,3,9,Sensor A,-37.81,144.96\n"],
    )

    report = validate_hourly_counts_snapshot(snapshot_path)

    assert report.passed is False
    assert any(issue.code == "SCHEMA_INVALID" for issue in report.errors)
    assert any(issue.code == "DIRECTION_TOTAL_MISMATCH" for issue in report.errors)


def test_hourly_count_snapshot_warns_for_duplicate_sensor_hour_and_incomplete_coverage(tmp_path):
    snapshot_path = _write_csv(
        tmp_path,
        [
            "a,1,2025-01-01,0,2,3,5,Sensor A,-37.81,144.96\n",
            "b,1,2025-01-01,0,1,1,2,Sensor A,-37.81,144.96\n",
        ],
    )

    report = validate_hourly_counts_snapshot(snapshot_path)

    assert report.passed is True
    assert any(issue.code == "DUPLICATE_SENSOR_HOUR" for issue in report.warnings)
    assert any(issue.code == "INCOMPLETE_HOUR_COVERAGE" for issue in report.warnings)
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/validation/test_hourly_counts.py -v
```

Expected: collection fails because `urbanflow.validation.hourly_counts` does not exist.

- [ ] **Step 3: Implement hourly-count validation**

Create `src/urbanflow/validation/hourly_counts.py`:

```python
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
                message=f"Hourly-count schema validation failed: {len(exc.failure_cases)} failure cases",
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
                        message=f"{incomplete_groups} location-date groups have fewer than 24 observed hours",
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
```

- [ ] **Step 4: Run validation tests and commit**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/validation -v
```

Expected: all validation tests pass.

Commit:

```powershell
git add src/urbanflow/validation tests/unit/validation
git commit -m "feat: validate Melbourne raw snapshots"
```

## Task 5: CLI, Dependencies, and Documentation

**Files:**
- Create: `tests/unit/validation/test_cli.py`
- Create: `src/urbanflow/validation/cli.py`
- Create: `scripts/validate_snapshot.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/unit/validation/test_cli.py`:

```python
import json

from urbanflow.validation.cli import main


def test_validation_cli_returns_zero_for_passing_snapshot(tmp_path, capsys):
    snapshot_path = tmp_path / "records.json"
    snapshot_path.write_text(
        json.dumps(
            [
                {
                    "location_id": 1,
                    "sensor_description": "Bourke Street",
                    "sensor_name": "Sensor A",
                    "installation_date": None,
                    "status": "A",
                    "latitude": -37.81,
                    "longitude": 144.96,
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["sensor_locations", str(snapshot_path)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "sensor_locations"
    assert payload["passed"] is True
    assert payload["error_count"] == 0


def test_validation_cli_returns_one_for_validation_failure(tmp_path, capsys):
    snapshot_path = tmp_path / "records.csv"
    snapshot_path.write_text(
        "id,location_id,sensing_date,hourday,direction_1,direction_2,pedestriancount,sensor_name,location\n"
        "a,1,2025-01-01,24,2,3,9,Sensor A,-37.81,144.96\n",
        encoding="utf-8",
    )

    exit_code = main(["hourly_counts", str(snapshot_path)])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["error_count"] >= 1


def test_validation_cli_returns_two_for_read_failure(tmp_path, capsys):
    exit_code = main(["hourly_counts", str(tmp_path / "missing.csv")])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["error_count"] == 1


def test_validation_script_help(repository_root):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, repository_root / "scripts" / "validate_snapshot.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Validate a local UrbanFlow AU raw snapshot" in result.stdout
```

- [ ] **Step 2: Run CLI tests and confirm RED**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest tests/unit/validation/test_cli.py -v
```

Expected: collection fails because `urbanflow.validation.cli` does not exist.

- [ ] **Step 3: Implement CLI and script**

Create `src/urbanflow/validation/cli.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from urbanflow.validation.pipeline import ValidationPipelineError, validate_snapshot
from urbanflow.validation.reports import ValidationReport

READ_ERROR_CODES = {"SNAPSHOT_READ_ERROR"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a local UrbanFlow AU raw snapshot.")
    parser.add_argument("dataset", choices=("sensor_locations", "hourly_counts"))
    parser.add_argument("snapshot_path", type=Path)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Optional root directory for full JSON validation reports.",
    )
    return parser


def _summary(report: ValidationReport) -> dict[str, object]:
    return {
        "dataset": report.dataset,
        "snapshot_path": report.snapshot_path,
        "passed": report.passed,
        "row_count": report.row_count,
        "error_count": len(report.errors),
        "warning_count": len(report.warnings),
    }


def _exit_code(report: ValidationReport) -> int:
    if any(issue.code in READ_ERROR_CODES for issue in report.errors):
        return 2
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = validate_snapshot(
            args.dataset,
            args.snapshot_path,
            report_root=args.report_root,
        )
    except ValidationPipelineError as exc:
        parser.error(str(exc))
    print(json.dumps(_summary(report), sort_keys=True))
    return _exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
```

Create `scripts/validate_snapshot.py`:

```python
from urbanflow.validation.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Update runtime dependencies and README**

Modify `pyproject.toml` dependencies:

```toml
dependencies = [
    "httpx>=0.28,<1",
    "pandas>=2.1,<3",
    "pandera[pandas]>=0.24,<1",
    "tenacity>=9,<10",
]
```

Add this README section after the hourly-count ingestion section:

## Validate a local raw snapshot

After generating raw snapshots, validate them before downstream processing:

```powershell
$sensorSnapshot = Get-ChildItem data/raw/melbourne/sensor_locations -Filter records.json -Recurse | Select-Object -First 1
python scripts/validate_snapshot.py sensor_locations $sensorSnapshot.FullName

$hourlySnapshot = Get-ChildItem data/raw/melbourne/hourly_counts -Filter records.csv -Recurse | Select-Object -First 1
python scripts/validate_snapshot.py hourly_counts $hourlySnapshot.FullName
```

Use `--report-root reports/data_quality` to write the full JSON quality report.
The command exits with `0` for pass, `1` for validation failures, and `2` for
invalid input or unreadable snapshot files.

- [ ] **Step 5: Install project dependencies and run full quality gate**

Run:

```powershell
& ..\..\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m ruff check . --no-cache
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m ruff format --check .
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest
```

Expected: Ruff has no diagnostics, format check reports all files formatted, and pytest
passes.

- [ ] **Step 6: Commit CLI, dependencies, and docs**

Commit:

```powershell
git add pyproject.toml README.md src/urbanflow/validation/cli.py scripts/validate_snapshot.py tests/unit/validation/test_cli.py
git commit -m "feat: add snapshot validation CLI"
```

## Final Integration

- [ ] **Step 1: Re-run the full quality gate on the feature branch**

Run:

```powershell
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m ruff check . --no-cache
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m ruff format --check .
$env:PYTHONPATH='src'; & ..\..\.venv\Scripts\python.exe -m pytest
```

Expected: all commands pass.

- [ ] **Step 2: Merge to `main`, verify on `main`, push only `main`**

From the main checkout:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 fetch origin main
git merge --ff-only codex/data-validation
& .\.venv\Scripts\python.exe -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 push origin main
```

Expected: only `main` is pushed; no `codex/*` branch is pushed to GitHub.

- [ ] **Step 3: Clean local worktree and branch**

Verify the worktree path is under `D:\Github项目\UrbanFlow-AU\.worktrees`, then run:

```powershell
git worktree remove --force D:\Github项目\UrbanFlow-AU\.worktrees\data-validation
git worktree prune
git branch -d codex/data-validation
```

Expected: `git worktree list` shows only the main checkout, and `git status --short --branch`
shows `main...origin/main`.
