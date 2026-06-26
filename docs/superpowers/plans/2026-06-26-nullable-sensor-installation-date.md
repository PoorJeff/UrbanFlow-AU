# Nullable Sensor Installation Date Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Preserve valid sensor records whose source installation date is null while retaining strict location validation.

**Architecture:** Modify only the sensor-location normalizer’s typed contract. The existing pipeline, runner, snapshot, and manifest code continue to consume the same normalized dictionary shape, now with a nullable date value.

**Tech Stack:** Python 3.11+, dataclasses, pytest, Ruff, existing httpx runner.

---

## File Structure

- Modify: src/urbanflow/ingestion/sensor_locations.py — make the installation date optional and remove it from strict required fields.
- Modify: tests/unit/ingestion/test_sensor_locations.py — capture the live-source null-date case as a regression test.
- Verify: scripts/ingest_sensor_locations.py — run the existing command once against the official API after unit tests pass.

### Task 1: Capture the null-date source contract in a failing test

**Files:**
- Modify: tests/unit/ingestion/test_sensor_locations.py
- Reference: src/urbanflow/ingestion/sensor_locations.py

- [ ] **Step 1: Add the desired nullable-date behavior test**

~~~python
def test_parse_sensor_location_preserves_null_installation_date() -> None:
    record = {**SOURCE_RECORD, "installation_date": None}

    sensor = parse_sensor_location(record)

    assert sensor.installation_date is None
    assert sensor.to_dict()["installation_date"] is None
~~~

Keep the existing invalid-coordinate test unchanged; it proves location fields remain strict.

- [ ] **Step 2: Run the one test and confirm it fails under the old contract**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_sensor_locations.py::test_parse_sensor_location_preserves_null_installation_date -v
~~~

Expected: FAIL with SensorLocationParseError naming installation_date, because the current required-field loop rejects null.

### Task 2: Make the date nullable with the smallest contract change

**Files:**
- Modify: src/urbanflow/ingestion/sensor_locations.py
- Test: tests/unit/ingestion/test_sensor_locations.py

- [ ] **Step 1: Change the typed contract and required-field list**

Replace the relevant parts of the existing source with:

~~~python
REQUIRED_FIELDS = (
    "location_id",
    "sensor_description",
    "sensor_name",
    "status",
    "latitude",
    "longitude",
)
~~~

~~~python
@dataclass(frozen=True)
class SensorLocation:
    location_id: int
    sensor_description: str
    sensor_name: str
    installation_date: str | None
    status: str
    latitude: float
    longitude: float
    note: str | None = None
    location_type: str | None = None
    direction_1: str | None = None
    direction_2: str | None = None
    location: dict[str, Any] | None = None
~~~

In parse_sensor_location, replace the date assignment only:

~~~python
installation_date=_optional_str(record.get("installation_date")),
~~~

Do not relax coercion or validation for any other field.

- [ ] **Step 2: Run the focused sensor-location test file**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/ingestion/sensor_locations.py tests/unit/ingestion/test_sensor_locations.py --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/ingestion/sensor_locations.py tests/unit/ingestion/test_sensor_locations.py
& .\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_sensor_locations.py -v
~~~

Expected: Ruff has no diagnostics and all five sensor-location tests pass.

- [ ] **Step 3: Commit the behavior correction**

~~~powershell
git add src/urbanflow/ingestion/sensor_locations.py tests/unit/ingestion/test_sensor_locations.py
git commit -m "fix: allow null sensor installation dates"
~~~

### Task 3: Verify the complete runner against the real source

**Files:**
- Verify: src/urbanflow/ingestion/sensor_location_pipeline.py
- Verify: src/urbanflow/ingestion/sensor_location_cli.py
- Verify: scripts/ingest_sensor_locations.py

- [ ] **Step 1: Run the full automated quality gate**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
~~~

Expected: Ruff has no diagnostics and all tests pass without live network access.

- [ ] **Step 2: Run the live runner and inspect its result**

Run:

~~~powershell
& .\.venv\Scripts\python.exe scripts/ingest_sensor_locations.py
~~~

Expected: exit code 0; JSON stdout reports source_total_count 136, record_count 136, and paths under data/raw/melbourne/sensor_locations and data/manifests/sensor_locations.

- [ ] **Step 3: Confirm generated files remain ignored**

Run:

~~~powershell
git check-ignore data/raw data/manifests
git status --short --ignored data/raw data/manifests
~~~

Expected: both directories are ignored; the generated snapshot and manifest are never staged.

- [ ] **Step 4: Perform final release self-evaluation**

Verify all of the following from fresh output before merging:

- The only parser contract relaxation is installation_date becoming str | None.
- Existing strict coordinate and required-field tests still pass.
- The new test observed the old parser failure before the fix.
- Automated tests never called the live API.
- The live run produced the entire source dataset and ignored local artifacts.

If every item passes, merge the local codex branch into main, re-run the full quality gate on main, and push only origin/main.
