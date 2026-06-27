# Hourly Count Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded hourly-count ingestion command that downloads City of Melbourne CSV exports, writes immutable local snapshots, and records provenance manifests.

**Architecture:** Extend the existing ingestion package with CSV export support in the Melbourne API client, small hourly-count domain helpers, a pipeline that validates row counts before finalizing snapshots, and a testable CLI wrapper. Keep this slice as raw ingestion only; validation, database persistence, sensor selection, and Parquet conversion remain later slices.

**Tech Stack:** Python 3.11+, `httpx`, `tenacity`, stdlib `csv`/`datetime`/`tempfile`, pytest, Ruff

---

## File Structure

- Modify `src/urbanflow/ingestion/melbourne_api.py`
  - Add `DatasetRecordCount`, export URL construction, count-query support, and CSV export download support.
- Modify `tests/unit/ingestion/test_melbourne_api.py`
  - Cover count queries and CSV export downloads with `httpx.MockTransport`.
- Modify `src/urbanflow/ingestion/snapshots.py`
  - Add a generic immutable file-snapshot mover for CSV snapshots.
- Modify `src/urbanflow/ingestion/manifests.py`
  - Add optional manifest metadata while preserving the existing manifest contract.
- Modify `tests/unit/ingestion/test_snapshots_and_manifests.py`
  - Cover CSV snapshot moving and metadata serialization.
- Create `src/urbanflow/ingestion/hourly_counts.py`
  - Own hourly-count constants, date-range validation, source query construction, and CSV row counting.
- Create `tests/unit/ingestion/test_hourly_counts.py`
  - Cover date ranges, query generation, and CSV row counting.
- Create `src/urbanflow/ingestion/hourly_count_pipeline.py`
  - Orchestrate count query, CSV download, row-count check, final snapshot move, manifest writing, and typed result metadata.
- Create `tests/unit/ingestion/test_hourly_count_pipeline.py`
  - Cover success, empty source range, and row-count mismatch.
- Create `src/urbanflow/ingestion/hourly_count_cli.py`
  - Own CLI parsing, client lifetime, JSON summary output, and operational exit codes.
- Create `tests/unit/ingestion/test_hourly_count_cli.py`
  - Cover CLI success, invalid arguments, expected pipeline failures, and script help.
- Create `scripts/ingest_hourly_counts.py`
  - Thin script wrapper for local execution.
- Modify `src/urbanflow/ingestion/__init__.py`
  - Export the hourly-count pipeline result and function.
- Modify `README.md`
  - Document the hourly-count ingestion command and update the first planned slice status.

Use the project virtual environment from the main checkout when executing inside a `.worktrees/...` directory:

```powershell
$python = 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe'
```

---

### Task 1: Melbourne API count and CSV export support

**Files:**
- Modify: `src/urbanflow/ingestion/melbourne_api.py`
- Test: `tests/unit/ingestion/test_melbourne_api.py`

- [ ] **Step 1: Add failing API-client tests**

Append these tests to `tests/unit/ingestion/test_melbourne_api.py`:

```python
def test_count_records_uses_limit_zero_and_where_clause() -> None:
    requested_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_params.append(dict(request.url.params))
        return json_response({"total_count": 2295, "results": []})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    api_client = MelbourneApiClient(http_client=http_client)

    result = api_client.count_records(
        "pedestrian-counting-system-monthly-counts-per-hour",
        where="sensing_date = date'2025-01-01'",
    )

    assert requested_params == [
        {"limit": "0", "where": "sensing_date = date'2025-01-01'"}
    ]
    assert result.dataset == "pedestrian-counting-system-monthly-counts-per-hour"
    assert result.total_count == 2295
    assert result.source_url.endswith(
        "/pedestrian-counting-system-monthly-counts-per-hour/records"
    )


def test_count_records_rejects_non_integer_total_count() -> None:
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: json_response({"total_count": "2295", "results": []})
        )
    )
    api_client = MelbourneApiClient(http_client=http_client)

    with pytest.raises(MelbourneApiError, match="total_count"):
        api_client.count_records("pedestrian-counting-system-monthly-counts-per-hour")


def test_export_url_points_to_dataset_export_endpoint() -> None:
    api_client = MelbourneApiClient(base_url="https://example.test/datasets")

    assert (
        api_client.export_url(
            "pedestrian-counting-system-monthly-counts-per-hour",
            export_format="csv",
        )
        == "https://example.test/datasets/"
        "pedestrian-counting-system-monthly-counts-per-hour/exports/csv"
    )


def test_export_csv_streams_selected_columns_to_file(tmp_path) -> None:
    requested_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_params.append(dict(request.url.params))
        assert request.url.path.endswith(
            "/pedestrian-counting-system-monthly-counts-per-hour/exports/csv"
        )
        return httpx.Response(
            200,
            content=(
                "id,location_id,sensing_date\n"
                "51120250101,51,2025-01-01\n"
            ).encode("utf-8"),
            headers={"content-type": "text/csv; charset=utf-8"},
        )

    output_path = tmp_path / "records.csv"
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    api_client = MelbourneApiClient(http_client=http_client)

    api_client.export_csv(
        "pedestrian-counting-system-monthly-counts-per-hour",
        output_path=output_path,
        select=["id", "location_id", "sensing_date"],
        where="sensing_date = date'2025-01-01'",
    )

    assert requested_params == [
        {
            "delimiter": ",",
            "select": "id,location_id,sensing_date",
            "where": "sensing_date = date'2025-01-01'",
            "with_bom": "false",
        }
    ]
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "id,location_id,sensing_date",
        "51120250101,51,2025-01-01",
    ]
```

- [ ] **Step 2: Run the focused API tests and verify RED**

Run:

```powershell
& $python -m pytest tests/unit/ingestion/test_melbourne_api.py -v
```

Expected: FAIL because `DatasetRecordCount`, `count_records`, `export_url`, or `export_csv` does not exist yet.

- [ ] **Step 3: Implement the API-client support**

Modify `src/urbanflow/ingestion/melbourne_api.py` with these changes:

```python
from collections.abc import Sequence
```

Add this dataclass below `DatasetRecords`:

```python
@dataclass(frozen=True)
class DatasetRecordCount:
    dataset: str
    source_url: str
    total_count: int
```

Add these methods to `MelbourneApiClient`:

```python
    def export_url(self, dataset: str, *, export_format: str) -> str:
        if not dataset:
            raise ValueError("dataset must not be empty")
        if not export_format:
            raise ValueError("export_format must not be empty")
        return f"{self._base_url}/{dataset}/exports/{export_format}"

    def count_records(self, dataset: str, *, where: str | None = None) -> DatasetRecordCount:
        source_url = self.records_url(dataset)
        params: dict[str, str | int] = {"limit": 0}
        if where:
            params["where"] = where

        payload = self._fetch_records_query(source_url, params=params)
        total_count = payload["total_count"]
        if not isinstance(total_count, int):
            raise MelbourneApiError("API response field 'total_count' must be an integer")

        return DatasetRecordCount(
            dataset=dataset,
            source_url=source_url,
            total_count=total_count,
        )

    def export_csv(
        self,
        dataset: str,
        *,
        output_path: Path,
        select: Sequence[str],
        where: str | None = None,
    ) -> None:
        if not select:
            raise ValueError("select must contain at least one column")

        export_url = self.export_url(dataset, export_format="csv")
        params: dict[str, str] = {
            "delimiter": ",",
            "select": ",".join(select),
            "with_bom": "false",
        }
        if where:
            params["where"] = where

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._http_client.stream("GET", export_url, params=params) as response:
                response.raise_for_status()
                with output_path.open("wb") as output_file:
                    for chunk in response.iter_bytes():
                        output_file.write(chunk)
        except httpx.HTTPError as exc:
            raise MelbourneApiError(f"Melbourne API CSV export failed: {exc}") from exc
```

Replace `_fetch_page()` with a two-layer implementation:

```python
    def _fetch_page(self, source_url: str, *, limit: int, offset: int) -> dict[str, Any]:
        payload = self._fetch_records_query(
            source_url,
            params={"limit": limit, "offset": offset},
        )
        if "results" not in payload:
            raise MelbourneApiError("Melbourne API response is missing 'results'")
        return payload

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, MelbourneApiError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1.0),
        reraise=True,
    )
    def _fetch_records_query(
        self,
        source_url: str,
        *,
        params: dict[str, str | int],
    ) -> dict[str, Any]:
        try:
            response = self._http_client.get(source_url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MelbourneApiError(f"Melbourne API request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise MelbourneApiError("Melbourne API response was not valid JSON") from exc

        if not isinstance(payload, dict):
            raise MelbourneApiError("Melbourne API response must be a JSON object")
        if "total_count" not in payload:
            raise MelbourneApiError("Melbourne API response is missing 'total_count'")
        return payload
```

Also import `Path` near the top:

```python
from pathlib import Path
```

- [ ] **Step 4: Run focused API checks and verify GREEN**

Run:

```powershell
& $python -m ruff check src/urbanflow/ingestion/melbourne_api.py tests/unit/ingestion/test_melbourne_api.py --no-cache
& $python -m ruff format --check src/urbanflow/ingestion/melbourne_api.py tests/unit/ingestion/test_melbourne_api.py
& $python -m pytest tests/unit/ingestion/test_melbourne_api.py -v
```

Expected: Ruff has no diagnostics and all API tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src/urbanflow/ingestion/melbourne_api.py tests/unit/ingestion/test_melbourne_api.py
git commit -m "feat: support Melbourne CSV exports"
```

---

### Task 2: Generic CSV snapshot finalization and manifest metadata

**Files:**
- Modify: `src/urbanflow/ingestion/snapshots.py`
- Modify: `src/urbanflow/ingestion/manifests.py`
- Test: `tests/unit/ingestion/test_snapshots_and_manifests.py`

- [ ] **Step 1: Add failing snapshot and manifest tests**

Append these tests to `tests/unit/ingestion/test_snapshots_and_manifests.py`:

```python
from urbanflow.ingestion.snapshots import move_file_snapshot
```

```python
def test_move_file_snapshot_places_file_in_immutable_dataset_path(tmp_path) -> None:
    source_path = tmp_path / "download.tmp"
    source_path.write_text("id,location_id\n1,3\n", encoding="utf-8")

    snapshot_path = move_file_snapshot(
        source_path=source_path,
        root_dir=tmp_path / "raw",
        dataset="hourly_counts",
        extracted_at=EXTRACTED_AT,
        filename="records.csv",
    )

    assert snapshot_path.as_posix().endswith(
        "melbourne/hourly_counts/extracted_at=20260625T083005Z/records.csv"
    )
    assert snapshot_path.read_text(encoding="utf-8") == "id,location_id\n1,3\n"
    assert not source_path.exists()

    replacement_source = tmp_path / "replacement.tmp"
    replacement_source.write_text("id,location_id\n2,4\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        move_file_snapshot(
            source_path=replacement_source,
            root_dir=tmp_path / "raw",
            dataset="hourly_counts",
            extracted_at=EXTRACTED_AT,
            filename="records.csv",
        )


def test_write_manifest_includes_optional_metadata(tmp_path) -> None:
    snapshot_path = tmp_path / "records.csv"
    snapshot_path.write_text("id,location_id\n1,3\n", encoding="utf-8")

    manifest_path = write_manifest(
        root_dir=tmp_path / "manifests",
        dataset="hourly_counts",
        source_url="https://example.test/datasets/hourly/exports/csv",
        extracted_at=EXTRACTED_AT,
        record_count=1,
        source_total_count=1,
        snapshot_path=snapshot_path,
        metadata={
            "snapshot_format": "csv",
            "selected_columns": ["id", "location_id"],
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["metadata"] == {
        "snapshot_format": "csv",
        "selected_columns": ["id", "location_id"],
    }
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
& $python -m pytest tests/unit/ingestion/test_snapshots_and_manifests.py -v
```

Expected: FAIL because `move_file_snapshot` and the `metadata` parameter do not exist yet.

- [ ] **Step 3: Implement snapshot finalization and metadata**

In `src/urbanflow/ingestion/snapshots.py`, add:

```python
def move_file_snapshot(
    *,
    source_path: Path,
    root_dir: Path,
    dataset: str,
    extracted_at: datetime,
    filename: str,
) -> Path:
    timestamp = format_extracted_at(extracted_at)
    snapshot_path = root_dir / "melbourne" / dataset / f"extracted_at={timestamp}" / filename
    if snapshot_path.exists():
        raise FileExistsError(snapshot_path)

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.replace(snapshot_path)
    return snapshot_path
```

In `src/urbanflow/ingestion/manifests.py`, add this import:

```python
from typing import Any
```

Change the `write_manifest()` signature to include:

```python
    metadata: dict[str, Any] | None = None,
```

After constructing the `manifest` dictionary, add:

```python
    if metadata is not None:
        manifest["metadata"] = metadata
```

- [ ] **Step 4: Run focused checks and verify GREEN**

Run:

```powershell
& $python -m ruff check src/urbanflow/ingestion/snapshots.py src/urbanflow/ingestion/manifests.py tests/unit/ingestion/test_snapshots_and_manifests.py --no-cache
& $python -m ruff format --check src/urbanflow/ingestion/snapshots.py src/urbanflow/ingestion/manifests.py tests/unit/ingestion/test_snapshots_and_manifests.py
& $python -m pytest tests/unit/ingestion/test_snapshots_and_manifests.py -v
```

Expected: Ruff has no diagnostics and all snapshot/manifest tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src/urbanflow/ingestion/snapshots.py src/urbanflow/ingestion/manifests.py tests/unit/ingestion/test_snapshots_and_manifests.py
git commit -m "feat: support CSV snapshot manifests"
```

---

### Task 3: Hourly-count domain helpers

**Files:**
- Create: `src/urbanflow/ingestion/hourly_counts.py`
- Test: `tests/unit/ingestion/test_hourly_counts.py`

- [ ] **Step 1: Write failing hourly-count helper tests**

Create `tests/unit/ingestion/test_hourly_counts.py`:

```python
from datetime import date

import pytest

from urbanflow.ingestion.hourly_counts import (
    HOURLY_COUNT_COLUMNS,
    HourlyCountDateRange,
    HourlyCountIngestionError,
    build_hourly_counts_where,
    count_csv_data_rows,
    parse_iso_date,
    validate_date_range,
    year_date_range,
)


def test_year_date_range_expands_to_calendar_year() -> None:
    date_range = year_date_range(2025)

    assert date_range == HourlyCountDateRange(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
    )


def test_parse_iso_date_rejects_invalid_input() -> None:
    with pytest.raises(HourlyCountIngestionError, match="YYYY-MM-DD"):
        parse_iso_date("2025/01/01")


def test_validate_date_range_rejects_reversed_dates() -> None:
    with pytest.raises(HourlyCountIngestionError, match="start_date"):
        validate_date_range(date(2025, 1, 2), date(2025, 1, 1))


def test_build_hourly_counts_where_uses_inclusive_dates() -> None:
    date_range = HourlyCountDateRange(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )

    assert (
        build_hourly_counts_where(date_range)
        == "sensing_date >= date'2025-01-01' AND sensing_date <= date'2025-01-31'"
    )


def test_count_csv_data_rows_counts_rows_after_header(tmp_path) -> None:
    csv_path = tmp_path / "records.csv"
    csv_path.write_text(
        "id,location_id,sensing_date\n"
        "51120250101,51,2025-01-01\n"
        "45620250101,45,2025-01-01\n",
        encoding="utf-8",
    )

    assert count_csv_data_rows(csv_path) == 2


def test_count_csv_data_rows_rejects_empty_file(tmp_path) -> None:
    csv_path = tmp_path / "records.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(HourlyCountIngestionError, match="header"):
        count_csv_data_rows(csv_path)


def test_hourly_count_columns_preserve_source_order() -> None:
    assert HOURLY_COUNT_COLUMNS == (
        "id",
        "location_id",
        "sensing_date",
        "hourday",
        "direction_1",
        "direction_2",
        "pedestriancount",
        "sensor_name",
        "location",
    )
```

- [ ] **Step 2: Run helper tests and verify RED**

Run:

```powershell
& $python -m pytest tests/unit/ingestion/test_hourly_counts.py -v
```

Expected: FAIL because `urbanflow.ingestion.hourly_counts` does not exist yet.

- [ ] **Step 3: Implement hourly-count helpers**

Create `src/urbanflow/ingestion/hourly_counts.py`:

```python
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

HOURLY_COUNTS_SOURCE_DATASET = "pedestrian-counting-system-monthly-counts-per-hour"
HOURLY_COUNTS_SNAPSHOT_DATASET = "hourly_counts"
HOURLY_COUNT_COLUMNS = (
    "id",
    "location_id",
    "sensing_date",
    "hourday",
    "direction_1",
    "direction_2",
    "pedestriancount",
    "sensor_name",
    "location",
)


class HourlyCountIngestionError(ValueError):
    """Raised when hourly-count ingestion inputs or exports are unusable."""


@dataclass(frozen=True)
class HourlyCountDateRange:
    start_date: date
    end_date: date


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HourlyCountIngestionError(
            f"Date '{value}' must use YYYY-MM-DD format"
        ) from exc


def validate_date_range(start_date: date, end_date: date) -> HourlyCountDateRange:
    if start_date > end_date:
        raise HourlyCountIngestionError("start_date must be on or before end_date")
    return HourlyCountDateRange(start_date=start_date, end_date=end_date)


def year_date_range(year: int) -> HourlyCountDateRange:
    if year < 1900:
        raise HourlyCountIngestionError("year must be 1900 or later")
    return HourlyCountDateRange(
        start_date=date(year, 1, 1),
        end_date=date(year, 12, 31),
    )


def build_hourly_counts_where(date_range: HourlyCountDateRange) -> str:
    return (
        f"sensing_date >= date'{date_range.start_date.isoformat()}' "
        f"AND sensing_date <= date'{date_range.end_date.isoformat()}'"
    )


def count_csv_data_rows(csv_path: Path) -> int:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)
            if not header:
                raise HourlyCountIngestionError("CSV export is missing a header row")
            return sum(1 for _row in reader)
    except OSError as exc:
        raise HourlyCountIngestionError(f"CSV export could not be read: {exc}") from exc
```

- [ ] **Step 4: Run focused checks and verify GREEN**

Run:

```powershell
& $python -m ruff check src/urbanflow/ingestion/hourly_counts.py tests/unit/ingestion/test_hourly_counts.py --no-cache
& $python -m ruff format --check src/urbanflow/ingestion/hourly_counts.py tests/unit/ingestion/test_hourly_counts.py
& $python -m pytest tests/unit/ingestion/test_hourly_counts.py -v
```

Expected: Ruff has no diagnostics and all hourly-count helper tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add src/urbanflow/ingestion/hourly_counts.py tests/unit/ingestion/test_hourly_counts.py
git commit -m "feat: add hourly count helpers"
```

---

### Task 4: Hourly-count ingestion pipeline

**Files:**
- Create: `src/urbanflow/ingestion/hourly_count_pipeline.py`
- Modify: `src/urbanflow/ingestion/__init__.py`
- Test: `tests/unit/ingestion/test_hourly_count_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Create `tests/unit/ingestion/test_hourly_count_pipeline.py`:

```python
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from urbanflow.ingestion.hourly_count_pipeline import ingest_hourly_counts
from urbanflow.ingestion.hourly_counts import (
    HOURLY_COUNT_COLUMNS,
    HOURLY_COUNTS_SOURCE_DATASET,
    HourlyCountDateRange,
    HourlyCountIngestionError,
)
from urbanflow.ingestion.melbourne_api import DatasetRecordCount

EXTRACTED_AT = datetime(2026, 6, 27, 8, 0, 5, tzinfo=UTC)
DATE_RANGE = HourlyCountDateRange(
    start_date=date(2025, 1, 1),
    end_date=date(2025, 1, 1),
)
CSV_BYTES = (
    "id,location_id,sensing_date,hourday,direction_1,direction_2,"
    "pedestriancount,sensor_name,location\n"
    '51120250101,51,2025-01-01,1,100,79,179,Fra118_T,"-37.8, 144.9"\n'
).encode("utf-8")


class FakeHourlyApiClient:
    def __init__(self, *, total_count: int, csv_bytes: bytes = CSV_BYTES) -> None:
        self.total_count = total_count
        self.csv_bytes = csv_bytes
        self.count_calls: list[tuple[str, str | None]] = []
        self.export_calls: list[tuple[str, tuple[str, ...], str | None, Path]] = []

    def count_records(self, dataset: str, *, where: str | None = None) -> DatasetRecordCount:
        self.count_calls.append((dataset, where))
        return DatasetRecordCount(
            dataset=dataset,
            source_url=f"https://example.test/{dataset}/records",
            total_count=self.total_count,
        )

    def export_url(self, dataset: str, *, export_format: str) -> str:
        return f"https://example.test/{dataset}/exports/{export_format}"

    def export_csv(
        self,
        dataset: str,
        *,
        output_path: Path,
        select: tuple[str, ...],
        where: str | None = None,
    ) -> None:
        self.export_calls.append((dataset, select, where, output_path))
        output_path.write_bytes(self.csv_bytes)


def test_ingest_hourly_counts_writes_snapshot_manifest_and_returns_metadata(
    tmp_path: Path,
) -> None:
    fake_client = FakeHourlyApiClient(total_count=1)

    result = ingest_hourly_counts(
        api_client=fake_client,
        raw_root_dir=tmp_path / "raw",
        manifest_root_dir=tmp_path / "manifests",
        date_range=DATE_RANGE,
        extracted_at=EXTRACTED_AT,
    )

    expected_where = "sensing_date >= date'2025-01-01' AND sensing_date <= date'2025-01-01'"
    assert fake_client.count_calls == [(HOURLY_COUNTS_SOURCE_DATASET, expected_where)]
    assert fake_client.export_calls[0][:3] == (
        HOURLY_COUNTS_SOURCE_DATASET,
        HOURLY_COUNT_COLUMNS,
        expected_where,
    )
    assert result.source_total_count == 1
    assert result.record_count == 1
    assert result.snapshot_path.read_bytes() == CSV_BYTES
    assert result.snapshot_path.as_posix().endswith(
        "melbourne/hourly_counts/extracted_at=20260627T080005Z/records.csv"
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "hourly_counts"
    assert manifest["record_count"] == 1
    assert manifest["source_total_count"] == 1
    assert manifest["metadata"] == {
        "date_range": {"end": "2025-01-01", "start": "2025-01-01"},
        "selected_columns": list(HOURLY_COUNT_COLUMNS),
        "sensor_filter": "all",
        "snapshot_format": "csv",
        "source_dataset": HOURLY_COUNTS_SOURCE_DATASET,
        "source_where": expected_where,
    }


def test_ingest_hourly_counts_rejects_empty_source_range(tmp_path: Path) -> None:
    fake_client = FakeHourlyApiClient(total_count=0)

    with pytest.raises(HourlyCountIngestionError, match="No hourly-count rows"):
        ingest_hourly_counts(
            api_client=fake_client,
            raw_root_dir=tmp_path / "raw",
            manifest_root_dir=tmp_path / "manifests",
            date_range=DATE_RANGE,
            extracted_at=EXTRACTED_AT,
        )

    assert fake_client.export_calls == []
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "manifests").exists()


def test_ingest_hourly_counts_rejects_count_mismatch_before_manifest(
    tmp_path: Path,
) -> None:
    fake_client = FakeHourlyApiClient(total_count=2)

    with pytest.raises(HourlyCountIngestionError, match="row count"):
        ingest_hourly_counts(
            api_client=fake_client,
            raw_root_dir=tmp_path / "raw",
            manifest_root_dir=tmp_path / "manifests",
            date_range=DATE_RANGE,
            extracted_at=EXTRACTED_AT,
        )

    assert list((tmp_path / "raw").rglob("records.csv")) == []
    assert not (tmp_path / "manifests").exists()
```

- [ ] **Step 2: Run pipeline tests and verify RED**

Run:

```powershell
& $python -m pytest tests/unit/ingestion/test_hourly_count_pipeline.py -v
```

Expected: FAIL because `urbanflow.ingestion.hourly_count_pipeline` does not exist yet.

- [ ] **Step 3: Implement the pipeline**

Create `src/urbanflow/ingestion/hourly_count_pipeline.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from urbanflow.ingestion.hourly_counts import (
    HOURLY_COUNT_COLUMNS,
    HOURLY_COUNTS_SNAPSHOT_DATASET,
    HOURLY_COUNTS_SOURCE_DATASET,
    HourlyCountDateRange,
    HourlyCountIngestionError,
    build_hourly_counts_where,
    count_csv_data_rows,
)
from urbanflow.ingestion.manifests import write_manifest
from urbanflow.ingestion.melbourne_api import DatasetRecordCount
from urbanflow.ingestion.snapshots import format_extracted_at, move_file_snapshot


class SupportsHourlyCountExport(Protocol):
    def count_records(self, dataset: str, *, where: str | None = None) -> DatasetRecordCount: ...

    def export_url(self, dataset: str, *, export_format: str) -> str: ...

    def export_csv(
        self,
        dataset: str,
        *,
        output_path: Path,
        select: tuple[str, ...],
        where: str | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class HourlyCountIngestionResult:
    source_dataset: str
    snapshot_dataset: str
    source_url: str
    extracted_at: datetime
    date_range: HourlyCountDateRange
    source_total_count: int
    record_count: int
    selected_columns: tuple[str, ...]
    snapshot_path: Path
    manifest_path: Path


def ingest_hourly_counts(
    *,
    api_client: SupportsHourlyCountExport,
    raw_root_dir: Path,
    manifest_root_dir: Path,
    date_range: HourlyCountDateRange,
    extracted_at: datetime | None = None,
) -> HourlyCountIngestionResult:
    extraction_time = extracted_at or datetime.now(UTC)
    source_where = build_hourly_counts_where(date_range)
    record_count_result = api_client.count_records(
        HOURLY_COUNTS_SOURCE_DATASET,
        where=source_where,
    )
    if record_count_result.total_count <= 0:
        raise HourlyCountIngestionError(
            "No hourly-count rows found for the requested date range"
        )

    timestamp = format_extracted_at(extraction_time)
    temp_dir = raw_root_dir / "melbourne" / HOURLY_COUNTS_SNAPSHOT_DATASET / "_tmp"
    temp_path = temp_dir / f"{timestamp}.records.csv.tmp"

    try:
        api_client.export_csv(
            HOURLY_COUNTS_SOURCE_DATASET,
            output_path=temp_path,
            select=HOURLY_COUNT_COLUMNS,
            where=source_where,
        )
        snapshot_record_count = count_csv_data_rows(temp_path)
        if snapshot_record_count != record_count_result.total_count:
            raise HourlyCountIngestionError(
                "CSV export row count "
                f"{snapshot_record_count} did not match source count "
                f"{record_count_result.total_count}"
            )

        snapshot_path = move_file_snapshot(
            source_path=temp_path,
            root_dir=raw_root_dir,
            dataset=HOURLY_COUNTS_SNAPSHOT_DATASET,
            extracted_at=extraction_time,
            filename="records.csv",
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

    source_url = api_client.export_url(HOURLY_COUNTS_SOURCE_DATASET, export_format="csv")
    manifest_path = write_manifest(
        root_dir=manifest_root_dir,
        dataset=HOURLY_COUNTS_SNAPSHOT_DATASET,
        source_url=source_url,
        extracted_at=extraction_time,
        record_count=snapshot_record_count,
        source_total_count=record_count_result.total_count,
        snapshot_path=snapshot_path,
        metadata={
            "date_range": {
                "end": date_range.end_date.isoformat(),
                "start": date_range.start_date.isoformat(),
            },
            "selected_columns": list(HOURLY_COUNT_COLUMNS),
            "sensor_filter": "all",
            "snapshot_format": "csv",
            "source_dataset": HOURLY_COUNTS_SOURCE_DATASET,
            "source_where": source_where,
        },
    )

    return HourlyCountIngestionResult(
        source_dataset=HOURLY_COUNTS_SOURCE_DATASET,
        snapshot_dataset=HOURLY_COUNTS_SNAPSHOT_DATASET,
        source_url=source_url,
        extracted_at=extraction_time,
        date_range=date_range,
        source_total_count=record_count_result.total_count,
        record_count=snapshot_record_count,
        selected_columns=HOURLY_COUNT_COLUMNS,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
    )
```

Modify `src/urbanflow/ingestion/__init__.py`:

```python
"""Data ingestion boundaries for UrbanFlow AU."""

from urbanflow.ingestion.hourly_count_pipeline import (
    HourlyCountIngestionResult,
    ingest_hourly_counts,
)
from urbanflow.ingestion.sensor_location_pipeline import (
    SensorLocationIngestionResult,
    ingest_sensor_locations,
)

__all__ = [
    "HourlyCountIngestionResult",
    "SensorLocationIngestionResult",
    "ingest_hourly_counts",
    "ingest_sensor_locations",
]
```

- [ ] **Step 4: Run focused checks and verify GREEN**

Run:

```powershell
& $python -m ruff check src/urbanflow/ingestion/hourly_count_pipeline.py src/urbanflow/ingestion/__init__.py tests/unit/ingestion/test_hourly_count_pipeline.py --no-cache
& $python -m ruff format --check src/urbanflow/ingestion/hourly_count_pipeline.py src/urbanflow/ingestion/__init__.py tests/unit/ingestion/test_hourly_count_pipeline.py
& $python -m pytest tests/unit/ingestion/test_hourly_count_pipeline.py -v
```

Expected: Ruff has no diagnostics and all hourly-count pipeline tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add src/urbanflow/ingestion/__init__.py src/urbanflow/ingestion/hourly_count_pipeline.py tests/unit/ingestion/test_hourly_count_pipeline.py
git commit -m "feat: add hourly count ingestion pipeline"
```

---

### Task 5: Hourly-count CLI, script, and README

**Files:**
- Create: `src/urbanflow/ingestion/hourly_count_cli.py`
- Create: `scripts/ingest_hourly_counts.py`
- Create: `tests/unit/ingestion/test_hourly_count_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/unit/ingestion/test_hourly_count_cli.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

from urbanflow.ingestion.hourly_count_cli import main
from urbanflow.ingestion.melbourne_api import DatasetRecordCount

CSV_BYTES = (
    "id,location_id,sensing_date,hourday,direction_1,direction_2,"
    "pedestriancount,sensor_name,location\n"
    '51120250101,51,2025-01-01,1,100,79,179,Fra118_T,"-37.8, 144.9"\n'
).encode("utf-8")


class FakeHourlyApiClient:
    def count_records(self, dataset: str, *, where: str | None = None) -> DatasetRecordCount:
        return DatasetRecordCount(
            dataset=dataset,
            source_url=f"https://example.test/{dataset}/records",
            total_count=1,
        )

    def export_url(self, dataset: str, *, export_format: str) -> str:
        return f"https://example.test/{dataset}/exports/{export_format}"

    def export_csv(
        self,
        dataset: str,
        *,
        output_path: Path,
        select: tuple[str, ...],
        where: str | None = None,
    ) -> None:
        output_path.write_bytes(CSV_BYTES)


def test_main_accepts_year_and_prints_json_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        ["--year", "2025"],
        api_client_factory=lambda http_client: FakeHourlyApiClient(),
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["date_range"] == {"end": "2025-12-31", "start": "2025-01-01"}
    assert summary["record_count"] == 1
    assert summary["source_total_count"] == 1
    assert (tmp_path / "data" / "raw").exists()
    assert (tmp_path / "data" / "manifests").exists()


def test_main_accepts_explicit_date_range(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"

    exit_code = main(
        [
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-01",
            "--raw-root",
            str(raw_root),
            "--manifest-root",
            str(manifest_root),
        ],
        api_client_factory=lambda http_client: FakeHourlyApiClient(),
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["date_range"] == {"end": "2025-01-01", "start": "2025-01-01"}
    assert Path(summary["snapshot_path"]).is_relative_to(raw_root)
    assert Path(summary["manifest_path"]).is_relative_to(manifest_root)


def test_main_rejects_missing_end_date(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--start-date", "2025-01-01"])

    assert exc_info.value.code == 2
    assert "provide both --start-date and --end-date" in capsys.readouterr().err


def test_main_rejects_invalid_date_format(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--start-date", "2025/01/01", "--end-date", "2025-01-01"])

    assert exc_info.value.code == 2
    assert "YYYY-MM-DD" in capsys.readouterr().err


def test_main_reports_pipeline_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class EmptyApiClient(FakeHourlyApiClient):
        def count_records(
            self,
            dataset: str,
            *,
            where: str | None = None,
        ) -> DatasetRecordCount:
            return DatasetRecordCount(
                dataset=dataset,
                source_url=f"https://example.test/{dataset}/records",
                total_count=0,
            )

    exit_code = main(
        [
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-01",
            "--raw-root",
            str(tmp_path / "raw"),
            "--manifest-root",
            str(tmp_path / "manifests"),
        ],
        api_client_factory=lambda http_client: EmptyApiClient(),
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error: No hourly-count rows found" in captured.err


def test_runner_script_displays_help() -> None:
    repository_root = Path(__file__).parents[3]

    completed_process = subprocess.run(
        [sys.executable, repository_root / "scripts" / "ingest_hourly_counts.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0
    assert "--year" in completed_process.stdout
    assert "--start-date" in completed_process.stdout
    assert "--end-date" in completed_process.stdout
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```powershell
& $python -m pytest tests/unit/ingestion/test_hourly_count_cli.py -v
```

Expected: FAIL because `urbanflow.ingestion.hourly_count_cli` and `scripts/ingest_hourly_counts.py` do not exist yet.

- [ ] **Step 3: Implement the CLI module**

Create `src/urbanflow/ingestion/hourly_count_cli.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from urbanflow.ingestion.hourly_count_pipeline import (
    HourlyCountIngestionResult,
    SupportsHourlyCountExport,
    ingest_hourly_counts,
)
from urbanflow.ingestion.hourly_counts import (
    HourlyCountDateRange,
    HourlyCountIngestionError,
    parse_iso_date,
    validate_date_range,
    year_date_range,
)
from urbanflow.ingestion.melbourne_api import MelbourneApiClient, MelbourneApiError


def positive_year(value: str) -> int:
    try:
        year = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("year must be an integer") from exc
    if year < 1900:
        raise argparse.ArgumentTypeError("year must be 1900 or later")
    return year


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Melbourne hourly pedestrian counts.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--year", type=positive_year)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser


def date_range_from_args(args: argparse.Namespace) -> HourlyCountDateRange:
    has_year = args.year is not None
    has_start = args.start_date is not None
    has_end = args.end_date is not None
    if has_year and (has_start or has_end):
        raise argparse.ArgumentTypeError(
            "provide either --year or --start-date/--end-date, not both"
        )
    if has_year:
        return year_date_range(args.year)
    if has_start != has_end:
        raise argparse.ArgumentTypeError("provide both --start-date and --end-date")
    if not has_start:
        raise argparse.ArgumentTypeError(
            "provide --year or both --start-date and --end-date"
        )
    return validate_date_range(parse_iso_date(args.start_date), parse_iso_date(args.end_date))


def result_summary(result: HourlyCountIngestionResult) -> dict[str, int | str | dict[str, str]]:
    return {
        "date_range": {
            "end": result.date_range.end_date.isoformat(),
            "start": result.date_range.start_date.isoformat(),
        },
        "extracted_at": result.extracted_at.isoformat(),
        "manifest_path": result.manifest_path.as_posix(),
        "record_count": result.record_count,
        "snapshot_path": result.snapshot_path.as_posix(),
        "source_dataset": result.source_dataset,
        "source_total_count": result.source_total_count,
        "source_url": result.source_url,
    }


def _default_api_client_factory(http_client: httpx.Client) -> MelbourneApiClient:
    return MelbourneApiClient(http_client=http_client)


def main(
    argv: Sequence[str] | None = None,
    *,
    api_client_factory: Callable[
        [httpx.Client], SupportsHourlyCountExport
    ] = _default_api_client_factory,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        date_range = date_range_from_args(args)
    except (argparse.ArgumentTypeError, HourlyCountIngestionError) as exc:
        parser.error(str(exc))

    try:
        with httpx.Client(timeout=30.0) as http_client:
            result = ingest_hourly_counts(
                api_client=api_client_factory(http_client),
                raw_root_dir=args.raw_root,
                manifest_root_dir=args.manifest_root,
                date_range=date_range,
            )
    except (HourlyCountIngestionError, MelbourneApiError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result_summary(result), sort_keys=True))
    return 0
```

- [ ] **Step 4: Add the script wrapper and README section**

Create `scripts/ingest_hourly_counts.py`:

```python
from urbanflow.ingestion.hourly_count_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

Add this README section after the sensor-location ingestion section:

````markdown
## Run hourly-count ingestion locally

```powershell
python scripts/ingest_hourly_counts.py --year 2025
```

The command downloads a bounded City of Melbourne hourly-count CSV export and
prints a JSON summary. Use `--year YYYY` for a full calendar year, or provide
both `--start-date YYYY-MM-DD` and `--end-date YYYY-MM-DD` for a smaller range.
There is no unbounded default because the source has million-row scale. By
default the command writes an immutable CSV snapshot below `data/raw/` and a
matching manifest below `data/manifests/`; both are ignored by Git.
````

Update the first planned delivery slice to:

```markdown
1. Melbourne sensor and hourly-count ingestion with immutable snapshots and manifests. Sensor-location ingestion is runnable locally; hourly-count ingestion has a bounded CSV export pipeline.
```

- [ ] **Step 5: Run CLI checks and verify GREEN**

Run:

```powershell
& $python -m ruff check src/urbanflow/ingestion/hourly_count_cli.py scripts/ingest_hourly_counts.py tests/unit/ingestion/test_hourly_count_cli.py --no-cache
& $python -m ruff format --check src/urbanflow/ingestion/hourly_count_cli.py scripts/ingest_hourly_counts.py tests/unit/ingestion/test_hourly_count_cli.py
& $python -m pytest tests/unit/ingestion/test_hourly_count_cli.py -v
& $python scripts/ingest_hourly_counts.py --help
```

Expected: Ruff has no diagnostics, CLI tests pass, and script help exits with code `0`.

- [ ] **Step 6: Commit Task 5**

Run:

```powershell
git add src/urbanflow/ingestion/hourly_count_cli.py scripts/ingest_hourly_counts.py tests/unit/ingestion/test_hourly_count_cli.py README.md
git commit -m "docs: document hourly count runner"
```

---

### Task 6: Full verification and live smoke test

**Files:**
- Verify: all project files
- Verify live command: `scripts/ingest_hourly_counts.py`

- [ ] **Step 1: Run the full local quality gate**

Run:

```powershell
& $python -m ruff check . --no-cache
& $python -m ruff format --check .
& $python -m pytest
```

Expected: Ruff has no diagnostics, formatting is clean, and all tests pass.

- [ ] **Step 2: Run a small live smoke test**

Run a one-day extraction so the first live check is fast and bounded:

```powershell
& $python scripts/ingest_hourly_counts.py --start-date 2025-01-01 --end-date 2025-01-01
```

Expected: exit code `0`; stdout is JSON with `record_count` equal to `source_total_count`, `date_range.start` equal to `2025-01-01`, `date_range.end` equal to `2025-01-01`, and paths below `data/raw/melbourne/hourly_counts/` and `data/manifests/hourly_counts/`.

- [ ] **Step 3: Inspect generated local artifacts without committing them**

Run:

```powershell
git status --short
```

Expected: generated files under `data/raw/` and `data/manifests/` do not appear because those paths are ignored by Git.

- [ ] **Step 4: Commit final implementation state if Task 6 changed tracked files**

If Task 6 reveals a tracked documentation correction, commit it:

```powershell
git add README.md docs/superpowers/plans/2026-06-27-hourly-count-ingestion.md
git commit -m "docs: refine hourly count ingestion notes"
```

If no tracked files changed, skip this commit.

- [ ] **Step 5: Merge to main, reverify, and push only main**

From the main checkout:

```powershell
git checkout main
git merge --ff-only codex/hourly-count-ingestion
& .\.venv\Scripts\python.exe -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
git push origin main
```

Expected: merge succeeds, full checks pass on `main`, and only `main` is pushed.

---

## Plan Self-Review

- Spec coverage: covered CSV export extraction, explicit bounded date range, dynamic source count, immutable CSV snapshot, metadata manifest, network-free tests, local CLI, README, and live small-range smoke test.
- Scope check: no Pandera, database, Prefect, sensor ranking, weather, Parquet, feature engineering, modeling, dashboard, or monitoring work is included.
- Type consistency: `HourlyCountDateRange`, `HourlyCountIngestionError`, `DatasetRecordCount`, `SupportsHourlyCountExport`, and `HourlyCountIngestionResult` are named consistently across tasks.
- Workflow fit: implementation should happen on a local `codex/hourly-count-ingestion` branch/worktree, merge into `main`, reverify on `main`, and push only `main`.
