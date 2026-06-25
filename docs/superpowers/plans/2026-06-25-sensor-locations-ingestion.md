# Sensor Locations Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a tested ingestion slice that fetches City of Melbourne sensor-location records, normalizes them, writes an immutable JSON snapshot, and creates a manifest with provenance metadata.

**Architecture:** Add a small `urbanflow.ingestion` package split by responsibility: `melbourne_api.py` for dataset-records pagination, `sensor_locations.py` for source-record normalization, `snapshots.py` for deterministic immutable JSON snapshots, and `manifests.py` for SHA-256 provenance manifests. Tests use fixed in-memory responses and temporary directories so CI remains network-independent.

**Tech Stack:** Python 3.11+, httpx, tenacity, pytest, Ruff

---

## File structure

- Modify `pyproject.toml`
  - Add runtime dependencies `httpx` and `tenacity`.
- Create `src/urbanflow/ingestion/__init__.py`
  - Public ingestion package boundary.
- Create `src/urbanflow/ingestion/melbourne_api.py`
  - Dataset-records URL construction, bounded-retry HTTP fetching, pagination, and API response validation.
- Create `src/urbanflow/ingestion/sensor_locations.py`
  - `SensorLocation` dataclass, source-record parser, coordinate validation, and a convenience `normalize_sensor_locations()` helper.
- Create `src/urbanflow/ingestion/snapshots.py`
  - Deterministic JSON snapshot writer and extracted-at formatting helper.
- Create `src/urbanflow/ingestion/manifests.py`
  - Snapshot hashing and manifest writer.
- Create `tests/unit/ingestion/test_melbourne_api.py`
  - API URL, pagination, and malformed-response tests.
- Create `tests/unit/ingestion/test_sensor_locations.py`
  - Sensor-location parser success and validation tests.
- Create `tests/unit/ingestion/test_snapshots_and_manifests.py`
  - Snapshot determinism, immutability, and manifest metadata tests.

### Task 1: Melbourne API pagination boundary

**Files:**
- Modify: `pyproject.toml`
- Create: `src/urbanflow/ingestion/__init__.py`
- Create: `src/urbanflow/ingestion/melbourne_api.py`
- Create: `tests/unit/ingestion/test_melbourne_api.py`

- [ ] **Step 1: Add the failing pagination tests**

Create `tests/unit/ingestion/test_melbourne_api.py` with:

```python
import json

import httpx
import pytest

from urbanflow.ingestion.melbourne_api import MelbourneApiClient, MelbourneApiError


def json_response(payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def test_fetch_all_records_paginates_until_source_total_count() -> None:
    requested_offsets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_offsets.append(int(request.url.params["offset"]))
        assert request.url.params["limit"] == "2"
        if request.url.params["offset"] == "0":
            return json_response({"total_count": 3, "results": [{"location_id": 1}, {"location_id": 2}]})
        return json_response({"total_count": 3, "results": [{"location_id": 3}]})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    api_client = MelbourneApiClient(http_client=http_client)

    page = api_client.fetch_all_records("pedestrian-counting-system-sensor-locations", limit=2)

    assert requested_offsets == [0, 2]
    assert page.total_count == 3
    assert page.records == [{"location_id": 1}, {"location_id": 2}, {"location_id": 3}]
    assert page.source_url.endswith("/pedestrian-counting-system-sensor-locations/records")


def test_fetch_all_records_rejects_malformed_payload_without_results() -> None:
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: json_response({"total_count": 1}))
    )
    api_client = MelbourneApiClient(http_client=http_client)

    with pytest.raises(MelbourneApiError, match="results"):
        api_client.fetch_all_records("pedestrian-counting-system-sensor-locations", limit=100)
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_melbourne_api.py -v
```

Expected: collection fails with `ModuleNotFoundError` for `urbanflow.ingestion` or `ImportError` for `MelbourneApiClient`.

- [ ] **Step 3: Add runtime dependencies**

Modify `pyproject.toml` dependencies to:

```toml
dependencies = [
    "httpx>=0.28,<1",
    "tenacity>=9,<10",
]
```

Then install the current package:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

- [ ] **Step 4: Add the minimal API client implementation**

Create `src/urbanflow/ingestion/__init__.py`:

```python
"""Data ingestion boundaries for UrbanFlow AU."""
```

Create `src/urbanflow/ingestion/melbourne_api.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

MELBOURNE_API_BASE_URL = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"


class MelbourneApiError(RuntimeError):
    """Raised when the Melbourne Open Data API cannot return a usable records page."""


@dataclass(frozen=True)
class DatasetRecords:
    dataset: str
    source_url: str
    total_count: int
    records: list[dict[str, Any]]


class MelbourneApiClient:
    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        base_url: str = MELBOURNE_API_BASE_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._http_client = http_client or httpx.Client(timeout=timeout_seconds)
        self._base_url = base_url.rstrip("/")

    def records_url(self, dataset: str) -> str:
        if not dataset:
            raise ValueError("dataset must not be empty")
        return f"{self._base_url}/{dataset}/records"

    def fetch_all_records(self, dataset: str, *, limit: int = 100) -> DatasetRecords:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        source_url = self.records_url(dataset)
        offset = 0
        total_count: int | None = None
        records: list[dict[str, Any]] = []

        while total_count is None or len(records) < total_count:
            payload = self._fetch_page(source_url, limit=limit, offset=offset)
            page_total_count = payload["total_count"]
            page_results = payload["results"]

            if not isinstance(page_total_count, int):
                raise MelbourneApiError("API response field 'total_count' must be an integer")
            if not isinstance(page_results, list):
                raise MelbourneApiError("API response field 'results' must be a list")
            if total_count is None:
                total_count = page_total_count
            if not page_results and len(records) < total_count:
                raise MelbourneApiError("API returned an empty page before total_count was reached")

            records.extend(page_results)
            offset += len(page_results)

        return DatasetRecords(
            dataset=dataset,
            source_url=source_url,
            total_count=total_count or 0,
            records=records,
        )

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, MelbourneApiError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1.0),
        reraise=True,
    )
    def _fetch_page(self, source_url: str, *, limit: int, offset: int) -> dict[str, Any]:
        try:
            response = self._http_client.get(source_url, params={"limit": limit, "offset": offset})
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
        if "results" not in payload:
            raise MelbourneApiError("Melbourne API response is missing 'results'")
        return payload
```

- [ ] **Step 5: Run the focused tests to verify GREEN**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_melbourne_api.py -v
```

Expected: both API tests pass.

- [ ] **Step 6: Commit the API boundary**

Run:

```powershell
git add pyproject.toml src/urbanflow/ingestion/__init__.py src/urbanflow/ingestion/melbourne_api.py tests/unit/ingestion/test_melbourne_api.py
git commit -m "feat: add Melbourne API pagination client"
```

### Task 2: Sensor-location parsing

**Files:**
- Create: `src/urbanflow/ingestion/sensor_locations.py`
- Create: `tests/unit/ingestion/test_sensor_locations.py`

- [ ] **Step 1: Add failing parser tests**

Create `tests/unit/ingestion/test_sensor_locations.py` with:

```python
import pytest

from urbanflow.ingestion.sensor_locations import (
    SensorLocationParseError,
    normalize_sensor_locations,
    parse_sensor_location,
)


SOURCE_RECORD = {
    "location_id": "3",
    "sensor_description": "Melbourne Central",
    "sensor_name": "Swa295_T",
    "installation_date": "2009-03-25",
    "note": None,
    "location_type": "Outdoor",
    "status": "A",
    "direction_1": "North",
    "direction_2": "South",
    "latitude": "-37.81101524",
    "longitude": "144.96429485",
    "location": {"lon": 144.96429485, "lat": -37.81101524},
}


def test_parse_sensor_location_normalizes_required_and_optional_fields() -> None:
    sensor = parse_sensor_location(SOURCE_RECORD)

    assert sensor.to_dict() == {
        "location_id": 3,
        "sensor_description": "Melbourne Central",
        "sensor_name": "Swa295_T",
        "installation_date": "2009-03-25",
        "status": "A",
        "latitude": -37.81101524,
        "longitude": 144.96429485,
        "note": None,
        "location_type": "Outdoor",
        "direction_1": "North",
        "direction_2": "South",
        "location": {"lon": 144.96429485, "lat": -37.81101524},
    }


def test_parse_sensor_location_rejects_missing_required_field() -> None:
    record = dict(SOURCE_RECORD)
    record.pop("sensor_name")

    with pytest.raises(SensorLocationParseError, match="sensor_name"):
        parse_sensor_location(record)


def test_parse_sensor_location_rejects_invalid_coordinates() -> None:
    record = dict(SOURCE_RECORD)
    record["latitude"] = -120

    with pytest.raises(SensorLocationParseError, match="latitude"):
        parse_sensor_location(record)


def test_normalize_sensor_locations_returns_json_ready_dicts() -> None:
    records = normalize_sensor_locations([SOURCE_RECORD])

    assert records == [parse_sensor_location(SOURCE_RECORD).to_dict()]
```

- [ ] **Step 2: Run parser tests to verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_sensor_locations.py -v
```

Expected: collection fails with `ModuleNotFoundError` or `ImportError` for `urbanflow.ingestion.sensor_locations`.

- [ ] **Step 3: Add the parser implementation**

Create `src/urbanflow/ingestion/sensor_locations.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUIRED_FIELDS = (
    "location_id",
    "sensor_description",
    "sensor_name",
    "installation_date",
    "status",
    "latitude",
    "longitude",
)


class SensorLocationParseError(ValueError):
    """Raised when a source sensor-location record cannot be normalized."""


@dataclass(frozen=True)
class SensorLocation:
    location_id: int
    sensor_description: str
    sensor_name: str
    installation_date: str
    status: str
    latitude: float
    longitude: float
    note: str | None = None
    location_type: str | None = None
    direction_1: str | None = None
    direction_2: str | None = None
    location: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "location_id": self.location_id,
            "sensor_description": self.sensor_description,
            "sensor_name": self.sensor_name,
            "installation_date": self.installation_date,
            "status": self.status,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "note": self.note,
            "location_type": self.location_type,
            "direction_1": self.direction_1,
            "direction_2": self.direction_2,
            "location": self.location,
        }


def parse_sensor_location(record: dict[str, Any]) -> SensorLocation:
    for field in REQUIRED_FIELDS:
        if field not in record or record[field] is None:
            raise SensorLocationParseError(f"Sensor location record is missing required field '{field}'")

    location_id = _coerce_int(record["location_id"], "location_id")
    latitude = _coerce_float(record["latitude"], "latitude")
    longitude = _coerce_float(record["longitude"], "longitude")
    _validate_coordinates(latitude=latitude, longitude=longitude)

    return SensorLocation(
        location_id=location_id,
        sensor_description=str(record["sensor_description"]),
        sensor_name=str(record["sensor_name"]),
        installation_date=str(record["installation_date"]),
        status=str(record["status"]),
        latitude=latitude,
        longitude=longitude,
        note=_optional_str(record.get("note")),
        location_type=_optional_str(record.get("location_type")),
        direction_1=_optional_str(record.get("direction_1")),
        direction_2=_optional_str(record.get("direction_2")),
        location=dict(record["location"]) if isinstance(record.get("location"), dict) else None,
    )


def normalize_sensor_locations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [parse_sensor_location(record).to_dict() for record in records]


def _coerce_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SensorLocationParseError(f"Field '{field}' must be an integer") from exc


def _coerce_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SensorLocationParseError(f"Field '{field}' must be numeric") from exc


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _validate_coordinates(*, latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise SensorLocationParseError("Field 'latitude' must be between -90 and 90")
    if not -180 <= longitude <= 180:
        raise SensorLocationParseError("Field 'longitude' must be between -180 and 180")
```

- [ ] **Step 4: Run parser tests to verify GREEN**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_sensor_locations.py -v
```

Expected: four parser tests pass.

- [ ] **Step 5: Commit parser work**

Run:

```powershell
git add src/urbanflow/ingestion/sensor_locations.py tests/unit/ingestion/test_sensor_locations.py
git commit -m "feat: normalize Melbourne sensor locations"
```

### Task 3: Deterministic snapshots and manifests

**Files:**
- Create: `src/urbanflow/ingestion/snapshots.py`
- Create: `src/urbanflow/ingestion/manifests.py`
- Create: `tests/unit/ingestion/test_snapshots_and_manifests.py`

- [ ] **Step 1: Add failing snapshot and manifest tests**

Create `tests/unit/ingestion/test_snapshots_and_manifests.py` with:

```python
from datetime import UTC, datetime
import hashlib
import json

import pytest

from urbanflow.ingestion.manifests import write_manifest
from urbanflow.ingestion.snapshots import format_extracted_at, write_json_snapshot


EXTRACTED_AT = datetime(2026, 6, 25, 8, 30, 5, tzinfo=UTC)
RECORDS = [
    {
        "location_id": 3,
        "sensor_name": "Swa295_T",
        "sensor_description": "Melbourne Central",
        "installation_date": "2009-03-25",
        "status": "A",
        "latitude": -37.81101524,
        "longitude": 144.96429485,
        "note": None,
        "location_type": "Outdoor",
        "direction_1": "North",
        "direction_2": "South",
        "location": {"lat": -37.81101524, "lon": 144.96429485},
    }
]


def test_format_extracted_at_uses_utc_compact_timestamp() -> None:
    assert format_extracted_at(EXTRACTED_AT) == "20260625T083005Z"


def test_write_json_snapshot_is_deterministic_and_immutable(tmp_path) -> None:
    snapshot_path = write_json_snapshot(
        records=RECORDS,
        root_dir=tmp_path,
        dataset="sensor_locations",
        extracted_at=EXTRACTED_AT,
    )

    assert snapshot_path.as_posix().endswith(
        "melbourne/sensor_locations/extracted_at=20260625T083005Z/records.json"
    )
    assert json.loads(snapshot_path.read_text(encoding="utf-8")) == RECORDS
    assert snapshot_path.read_text(encoding="utf-8").endswith("\n")

    with pytest.raises(FileExistsError):
        write_json_snapshot(
            records=RECORDS,
            root_dir=tmp_path,
            dataset="sensor_locations",
            extracted_at=EXTRACTED_AT,
        )


def test_write_manifest_records_snapshot_hash_and_counts(tmp_path) -> None:
    snapshot_path = write_json_snapshot(
        records=RECORDS,
        root_dir=tmp_path / "raw",
        dataset="sensor_locations",
        extracted_at=EXTRACTED_AT,
    )

    manifest_path = write_manifest(
        root_dir=tmp_path / "manifests",
        dataset="sensor_locations",
        source_url="https://example.test/datasets/sensor_locations/records",
        extracted_at=EXTRACTED_AT,
        record_count=1,
        source_total_count=136,
        snapshot_path=snapshot_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

    assert manifest["schema_version"] == 1
    assert manifest["dataset"] == "sensor_locations"
    assert manifest["record_count"] == 1
    assert manifest["source_total_count"] == 136
    assert manifest["snapshot_sha256"] == expected_hash
    assert manifest["snapshot_path"] == snapshot_path.as_posix()
```

- [ ] **Step 2: Run snapshot tests to verify RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_snapshots_and_manifests.py -v
```

Expected: collection fails with `ImportError` for `urbanflow.ingestion.snapshots` or `urbanflow.ingestion.manifests`.

- [ ] **Step 3: Add snapshot and manifest implementations**

Create `src/urbanflow/ingestion/snapshots.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


def format_extracted_at(extracted_at: datetime) -> str:
    return extracted_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_json_snapshot(
    *,
    records: list[dict[str, Any]],
    root_dir: Path,
    dataset: str,
    extracted_at: datetime,
) -> Path:
    timestamp = format_extracted_at(extracted_at)
    snapshot_path = (
        root_dir / "melbourne" / dataset / f"extracted_at={timestamp}" / "records.json"
    )
    if snapshot_path.exists():
        raise FileExistsError(snapshot_path)

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_text = json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True)
    snapshot_path.write_text(f"{snapshot_text}\n", encoding="utf-8")
    return snapshot_path
```

Create `src/urbanflow/ingestion/manifests.py`:

```python
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

from urbanflow.ingestion.snapshots import format_extracted_at


def write_manifest(
    *,
    root_dir: Path,
    dataset: str,
    source_url: str,
    extracted_at: datetime,
    record_count: int,
    source_total_count: int,
    snapshot_path: Path,
) -> Path:
    timestamp = format_extracted_at(extracted_at)
    manifest_path = root_dir / dataset / f"{timestamp}.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "source_url": source_url,
        "extracted_at": timestamp,
        "record_count": record_count,
        "source_total_count": source_total_count,
        "snapshot_path": snapshot_path.as_posix(),
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    manifest_path.write_text(f"{manifest_text}\n", encoding="utf-8")
    return manifest_path
```

- [ ] **Step 4: Run snapshot and manifest tests to verify GREEN**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_snapshots_and_manifests.py -v
```

Expected: three tests pass.

- [ ] **Step 5: Commit snapshot and manifest work**

Run:

```powershell
git add src/urbanflow/ingestion/snapshots.py src/urbanflow/ingestion/manifests.py tests/unit/ingestion/test_snapshots_and_manifests.py
git commit -m "feat: write sensor location snapshots and manifests"
```

### Task 4: Quality gate and documentation touchpoint

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a README status note for the first ingestion slice**

Modify the “Planned delivery slices” section so item 1 reads:

```markdown
1. Melbourne sensor and hourly-count ingestion with immutable snapshots and manifests. Sensor-location ingestion is the first functional slice.
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

- [ ] **Step 3: Commit the README touchpoint**

Run:

```powershell
git add README.md
git commit -m "docs: note sensor location ingestion slice"
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
