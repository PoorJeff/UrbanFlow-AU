# Sensor Location Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Python function entry point that runs sensor-location fetching, normalization, snapshot writing, and manifest writing as one tested ingestion pipeline.

**Architecture:** Add `sensor_location_pipeline.py` as a thin orchestration layer over the existing Melbourne API client, sensor-location parser, snapshot writer, and manifest writer. Keep IO roots and the API client injected so tests remain deterministic and network-free.

**Tech Stack:** Python 3.11+, pytest, Ruff, existing `httpx`/`tenacity` ingestion dependencies

---

## File structure

- Create `src/urbanflow/ingestion/sensor_location_pipeline.py`
  - Define dataset constants, result dataclass, an API-client protocol, and `ingest_sensor_locations`.
- Create `tests/unit/ingestion/test_sensor_location_pipeline.py`
  - Cover happy-path orchestration and failure-before-output behavior.
- Modify `src/urbanflow/ingestion/__init__.py`
  - Export `SensorLocationIngestionResult` and `ingest_sensor_locations`.
- Modify `README.md`
  - Note that the first ingestion slice now has a callable Python entry point.

### Task 1: Pipeline orchestration function

**Files:**
- Create: `tests/unit/ingestion/test_sensor_location_pipeline.py`
- Create: `src/urbanflow/ingestion/sensor_location_pipeline.py`
- Modify: `src/urbanflow/ingestion/__init__.py`

- [ ] **Step 1: Write the failing pipeline tests**

Create `tests/unit/ingestion/test_sensor_location_pipeline.py`:

```python
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from urbanflow.ingestion.melbourne_api import DatasetRecords
from urbanflow.ingestion.sensor_location_pipeline import (
    SENSOR_LOCATIONS_SOURCE_DATASET,
    SensorLocationIngestionResult,
    ingest_sensor_locations,
)
from urbanflow.ingestion.sensor_locations import SensorLocationParseError


EXTRACTED_AT = datetime(2026, 6, 25, 9, 45, 0, tzinfo=UTC)
SOURCE_RECORD = {
    "location_id": 3,
    "sensor_description": "Melbourne Central",
    "sensor_name": "Swa295_T",
    "installation_date": "2009-03-25",
    "note": None,
    "location_type": "Outdoor",
    "status": "A",
    "direction_1": "North",
    "direction_2": "South",
    "latitude": -37.81101524,
    "longitude": 144.96429485,
    "location": {"lon": 144.96429485, "lat": -37.81101524},
}


class FakeApiClient:
    def __init__(self, records: list[dict[str, Any]], *, total_count: int = 136) -> None:
        self.records = records
        self.total_count = total_count
        self.calls: list[tuple[str, int]] = []

    def fetch_all_records(self, dataset: str, *, limit: int = 100) -> DatasetRecords:
        self.calls.append((dataset, limit))
        return DatasetRecords(
            dataset=dataset,
            source_url=f"https://example.test/{dataset}/records",
            total_count=self.total_count,
            records=self.records,
        )


def test_ingest_sensor_locations_writes_snapshot_manifest_and_returns_metadata(tmp_path: Path) -> None:
    api_client = FakeApiClient([SOURCE_RECORD], total_count=136)

    result = ingest_sensor_locations(
        api_client=api_client,
        raw_root_dir=tmp_path / "raw",
        manifest_root_dir=tmp_path / "manifests",
        extracted_at=EXTRACTED_AT,
        page_limit=50,
    )

    assert isinstance(result, SensorLocationIngestionResult)
    assert api_client.calls == [(SENSOR_LOCATIONS_SOURCE_DATASET, 50)]
    assert result.source_total_count == 136
    assert result.record_count == 1
    assert result.snapshot_path.exists()
    assert result.manifest_path.exists()

    snapshot = json.loads(result.snapshot_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert snapshot[0]["location_id"] == 3
    assert snapshot[0]["sensor_name"] == "Swa295_T"
    assert manifest["record_count"] == 1
    assert manifest["source_total_count"] == 136
    assert manifest["source_url"] == result.source_url
    assert manifest["snapshot_path"] == result.snapshot_path.as_posix()


def test_ingest_sensor_locations_fails_before_writing_outputs_for_invalid_record(
    tmp_path: Path,
) -> None:
    invalid_record = dict(SOURCE_RECORD)
    invalid_record["latitude"] = -120
    api_client = FakeApiClient([invalid_record])

    with pytest.raises(SensorLocationParseError):
        ingest_sensor_locations(
            api_client=api_client,
            raw_root_dir=tmp_path / "raw",
            manifest_root_dir=tmp_path / "manifests",
            extracted_at=EXTRACTED_AT,
        )

    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "manifests").exists()
```

- [ ] **Step 2: Run the pipeline tests to verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_sensor_location_pipeline.py -v
```

Expected: collection fails with `ModuleNotFoundError` or `ImportError` for `urbanflow.ingestion.sensor_location_pipeline`.

- [ ] **Step 3: Add the pipeline implementation**

Create `src/urbanflow/ingestion/sensor_location_pipeline.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from urbanflow.ingestion.melbourne_api import DatasetRecords
from urbanflow.ingestion.manifests import write_manifest
from urbanflow.ingestion.sensor_locations import normalize_sensor_locations
from urbanflow.ingestion.snapshots import write_json_snapshot

SENSOR_LOCATIONS_SOURCE_DATASET = "pedestrian-counting-system-sensor-locations"
SENSOR_LOCATIONS_SNAPSHOT_DATASET = "sensor_locations"


class SupportsDatasetRecords(Protocol):
    def fetch_all_records(self, dataset: str, *, limit: int = 100) -> DatasetRecords:
        ...


@dataclass(frozen=True)
class SensorLocationIngestionResult:
    source_dataset: str
    snapshot_dataset: str
    source_url: str
    extracted_at: datetime
    source_total_count: int
    record_count: int
    snapshot_path: Path
    manifest_path: Path


def ingest_sensor_locations(
    *,
    api_client: SupportsDatasetRecords,
    raw_root_dir: Path,
    manifest_root_dir: Path,
    extracted_at: datetime | None = None,
    page_limit: int = 100,
) -> SensorLocationIngestionResult:
    if page_limit <= 0:
        raise ValueError("page_limit must be greater than zero")

    extraction_time = extracted_at or datetime.now(UTC)
    dataset_records = api_client.fetch_all_records(
        SENSOR_LOCATIONS_SOURCE_DATASET,
        limit=page_limit,
    )
    normalized_records = normalize_sensor_locations(dataset_records.records)
    snapshot_path = write_json_snapshot(
        records=normalized_records,
        root_dir=raw_root_dir,
        dataset=SENSOR_LOCATIONS_SNAPSHOT_DATASET,
        extracted_at=extraction_time,
    )
    manifest_path = write_manifest(
        root_dir=manifest_root_dir,
        dataset=SENSOR_LOCATIONS_SNAPSHOT_DATASET,
        source_url=dataset_records.source_url,
        extracted_at=extraction_time,
        record_count=len(normalized_records),
        source_total_count=dataset_records.total_count,
        snapshot_path=snapshot_path,
    )

    return SensorLocationIngestionResult(
        source_dataset=SENSOR_LOCATIONS_SOURCE_DATASET,
        snapshot_dataset=SENSOR_LOCATIONS_SNAPSHOT_DATASET,
        source_url=dataset_records.source_url,
        extracted_at=extraction_time,
        source_total_count=dataset_records.total_count,
        record_count=len(normalized_records),
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
    )
```

- [ ] **Step 4: Export the public entry point**

Modify `src/urbanflow/ingestion/__init__.py`:

```python
"""Data ingestion boundaries for UrbanFlow AU."""

from urbanflow.ingestion.sensor_location_pipeline import (
    SensorLocationIngestionResult,
    ingest_sensor_locations,
)

__all__ = ["SensorLocationIngestionResult", "ingest_sensor_locations"]
```

- [ ] **Step 5: Run focused tests to verify GREEN**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_sensor_location_pipeline.py -v
```

Expected: two pipeline tests pass.

- [ ] **Step 6: Commit pipeline implementation**

Run:

```powershell
git add src/urbanflow/ingestion/__init__.py src/urbanflow/ingestion/sensor_location_pipeline.py tests/unit/ingestion/test_sensor_location_pipeline.py
git commit -m "feat: add sensor location ingestion pipeline"
```

### Task 2: Documentation and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README status note**

Modify the first planned delivery slice to:

```markdown
1. Melbourne sensor and hourly-count ingestion with immutable snapshots and manifests. Sensor-location ingestion now has a tested Python function entry point.
```

- [ ] **Step 2: Run the complete quality gate**

Run:

```powershell
& .\.venv\Scripts\python.exe -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
git status --short --branch
```

Expected: Ruff passes, formatting is already correct, pytest reports all tests passing, and Git shows only the README change.

- [ ] **Step 3: Commit README**

Run:

```powershell
git add README.md
git commit -m "docs: note sensor location pipeline entry point"
```

- [ ] **Step 4: Final branch verification**

Run:

```powershell
& .\.venv\Scripts\python.exe -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
git status --short --branch
```

Expected: all checks pass and the feature branch is clean.
