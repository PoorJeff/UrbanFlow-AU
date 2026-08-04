# Snapshot-First Supervised CSV Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, offline CLI that turns one manifest-verified City of Melbourne hourly-count snapshot into a safely written direct `1..24`-hour supervised CSV for the existing evaluation and LightGBM artifact commands.

**Architecture:** A small modeling module reads immutable source bytes exactly once, parses and validates that exact in-memory DataFrame, verifies the matching ingestion manifest, maps source fields through the established Melbourne-time conversion, then delegates all feature generation to `build_supervised_frame`. A thin CLI parses local paths and preserves the project’s existing `0`/`2`/`1` success, input-error, and write-error convention.

**Tech Stack:** Python 3.12, pandas, existing Pandera validation, `zoneinfo` Melbourne conversion, argparse, pytest, Ruff.

## Global Constraints

- The builder is fully offline: it must not access the Melbourne API, PostgreSQL, MLflow, a model artifact, or a Dashboard.
- Inputs are an hourly-count CSV, its schema-v1 ingestion manifest, and the existing explicit local `HolidayCalendar` JSON; no date/horizon/download option is added.
- The generated features must come solely from `build_supervised_frame` with its fixed direct horizons `1..24`; do not add recursive predictions, imputation of counts, or model fallbacks.
- Use the exact snapshot bytes for both SHA-256 provenance and the DataFrame that is validated and converted; reject a mismatched manifest before output creation.
- Preserve source `DUPLICATE_SOURCE_ID` and `INCOMPLETE_HOUR_COVERAGE` as warnings, but reject `DUPLICATE_SENSOR_HOUR` because the feature input contract forbids duplicate sensor-time rows.
- The holiday calendar must cover every generated `target_observed_at` local date, including the final 24 target hours.
- Output CSVs are new local files only: never overwrite an existing destination, write through a temporary sibling, validate timestamp round-trip, publish with same-directory non-overwriting `os.link`, and remove temporary files on failure.
- Generated data and artifacts remain under ignored `data/processed/` and `models/`; no snapshot, credential, artifact, or evaluation output is committed.
- CI and ordinary pytest remain network-, PostgreSQL-, MLflow-, and artifact-free.

---

## File structure

| Path | Change | Responsibility |
| --- | --- | --- |
| `src/urbanflow/modeling/supervised_dataset.py` | Create | Manifest verification, same-frame validation, canonical observation conversion, calendar coverage check, atomic CSV output, and result metadata. |
| `tests/unit/modeling/test_supervised_dataset.py` | Create | Cover provenance, conversion, warning/error policy, calendar coverage, timestamp round-trip, and write cleanup. |
| `src/urbanflow/modeling/supervised_dataset_cli.py` | Create | Argparse boundary, JSON serialization, and stable error-to-exit-code mapping. |
| `scripts/build_supervised_csv.py` | Create | Minimal executable wrapper. |
| `tests/unit/modeling/test_supervised_dataset_cli.py` | Create | CLI stdout/stderr/exit-code and wrapper-help coverage. |
| `README.md` | Modify | Document the local bridge and state that it is not data download, database loading, training, or serving. |

## Task 1: Core supervised dataset builder

**Files:**
- Create: `src/urbanflow/modeling/supervised_dataset.py`
- Create: `tests/unit/modeling/test_supervised_dataset.py`

**Interfaces:**
- Consumes: `validate_hourly_counts_frame`, `melbourne_observed_at`, `build_supervised_frame`, `HolidayCalendar`, and `read_supervised_csv`.
- Produces: `SupervisedSnapshotBuildError`, `SupervisedSnapshotWriteError`, `SupervisedSnapshotBuildResult`, and `build_supervised_csv_from_hourly_snapshot(snapshot_path, manifest_path, output_path, *, holiday_calendar)`.

- [ ] **Step 1: Write the failing core-builder tests**

Create helpers that write a valid City of Melbourne CSV, its matching schema-v1 manifest, and an explicit holiday calendar. The manifest fixture must carry all eight v1 fields: `schema_version`, `dataset`, `source_url`, `extracted_at`, `record_count`, `source_total_count`, `snapshot_path`, and lowercase `snapshot_sha256`. Use 200 complete hourly rows for one location so the expected output is `200 * 24 == 4800` rows and `4500` rows have non-missing targets.

```python
def write_hourly_snapshot(tmp_path: Path, *, periods: int = 200) -> Path:
    timestamps = pd.date_range("2025-04-01 00:00", periods=periods, freq="h")
    frame = pd.DataFrame(
        {
            "id": [f"101{stamp:%Y%m%d%H}" for stamp in timestamps],
            "location_id": [101] * periods,
            "sensing_date": timestamps.strftime("%Y-%m-%d"),
            "hourday": timestamps.hour,
            "direction_1": [4] * periods,
            "direction_2": [6] * periods,
            "pedestriancount": [10] * periods,
            "sensor_name": ["Demo sensor"] * periods,
            "location": ["-37.8, 144.9"] * periods,
        }
    )
    path = tmp_path / "records.csv"
    frame.to_csv(path, index=False)
    return path


def test_builder_writes_direct_rows_from_a_verified_snapshot(tmp_path: Path) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    output = tmp_path / "supervised.csv"

    result = build_supervised_csv_from_hourly_snapshot(
        snapshot, manifest, output, holiday_calendar=write_calendar(tmp_path)
    )

    round_tripped = read_supervised_csv(output)
    assert result.source_row_count == 200
    assert result.supervised_row_count == 4800
    assert result.training_row_count == 4500
    assert set(round_tripped["forecast_horizon"]) == set(range(1, 25))
    assert str(round_tripped["forecast_origin_at"].dtype) == "datetime64[ns, UTC]"
    assert round_tripped["temperature"].isna().all()
    assert round_tripped["temperature_missing"].all()
```

Add focused tests for all of the following:

- Spy on `Path.read_bytes` and `pandas.read_csv` to prove the builder reads and parses the hourly-count CSV exactly once; mutate the raw file after writing its manifest and assert the mismatch creates neither a temporary nor destination output.
- Reject non-object or malformed JSON, each missing required manifest field, wrong schema/dataset, blank `source_url` or stored `snapshot_path`, an invalid `extracted_at`, boolean/negative counts, invalid lowercase SHA format, and SHA/row-count mismatch. Every case must produce the stable mismatch error and no output. Accept unrelated extra/`metadata` fields, and do **not** compare `source_total_count` with `record_count`.
- Prove worktree relocation: a manifest whose stored `snapshot_path` is stale/deleted succeeds when the supplied snapshot bytes match; the implementation must not read or resolve that stored path.
- Cover hard schema and direction-total validation errors, header-only/empty frames, duplicate sensor-hour warning, a missing source hour that produces a missing marker, and holiday coverage ending before the final target date.
- Prove an already existing destination remains byte-for-byte unchanged; simulate `to_csv`, round-trip, and `os.link` failures and verify no temporary sibling remains. Simulate a destination appearing at publish time and assert it is not overwritten.

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/modeling/test_supervised_dataset.py -v
```

Expected: FAIL because `urbanflow.modeling.supervised_dataset` does not exist.

- [ ] **Step 2: Implement the core builder and its precise error policy**

Create `src/urbanflow/modeling/supervised_dataset.py` with these public types and imports.

```python
class SupervisedSnapshotBuildError(ValueError):
    """Raised when a snapshot cannot safely become supervised rows."""


class SupervisedSnapshotWriteError(RuntimeError):
    """Raised when valid supervised rows cannot be written safely."""


@dataclass(frozen=True, slots=True)
class SupervisedSnapshotBuildResult:
    snapshot_path: Path
    manifest_path: Path
    output_path: Path
    source_row_count: int
    supervised_row_count: int
    training_row_count: int
    validation_warning_count: int
    snapshot_sha256: str
```

Keep raw-byte and manifest helpers private to this new module; do not modify
`snapshot_readers.py`, reuse `sha256_file`, or call a validator that re-reads a
path. Implement `build_supervised_csv_from_hourly_snapshot` in this exact order:

```python
if os.path.lexists(output_path):
    raise SupervisedSnapshotBuildError(
        f"supervised CSV output already exists: {output_path}"
    )
manifest = _load_hourly_count_manifest(manifest_path)
frame, snapshot_sha256 = _read_hourly_counts_snapshot_once(snapshot_path)
_verify_hourly_count_manifest(
    manifest, snapshot_sha256=snapshot_sha256, source_row_count=len(frame)
)
if frame.empty:
    raise SupervisedSnapshotBuildError("hourly-count snapshot contains no rows")
report = validate_hourly_counts_frame(frame, snapshot_path=snapshot_path)
if report.errors:
    codes = ", ".join(issue.code for issue in report.errors)
    raise SupervisedSnapshotBuildError(
        f"hourly-count snapshot validation failed: {codes}"
    )
if any(issue.code == "DUPLICATE_SENSOR_HOUR" for issue in report.warnings):
    raise SupervisedSnapshotBuildError("hourly-count snapshot contains duplicate sensor-hour rows")
observations = _observations_from_hourly_count_frame(frame)
supervised = build_supervised_frame(
    observations, public_holidays=holiday_calendar.public_holidays
)
_require_calendar_coverage(supervised, holiday_calendar)
_write_new_supervised_csv(supervised, output_path)
```

`_read_hourly_counts_snapshot_once(path)` must use one `path.read_bytes()` call,
calculate `hashlib.sha256(source_bytes).hexdigest()`, and parse those same bytes
using `pd.read_csv(BytesIO(source_bytes), dtype=str, keep_default_na=False)`.
Map `OSError`, `UnicodeError`, `pandas.errors.EmptyDataError`, and
`pandas.errors.ParserError` to `SupervisedSnapshotBuildError` before output work,
so missing, unreadable, empty, undecodable, or malformed source files are
operator input errors rather than CLI crashes.

`_load_hourly_count_manifest` must load UTF-8 JSON and map `OSError`,
`UnicodeError`, and `json.JSONDecodeError` to the stable manifest-mismatch
error.
`_verify_hourly_count_manifest` must require all eight v1 fields, reject a
non-object, require `schema_version` to be a plain non-boolean integer exactly
equal to `1`, require `dataset == "hourly_counts"`, require non-empty string
`source_url` and `snapshot_path`, parse `extracted_at` strictly with
`%Y%m%dT%H%M%SZ`, require non-boolean non-negative integer `record_count` and
`source_total_count`, and require a lowercase 64-hex `snapshot_sha256`. Compare
only `record_count` with the parsed frame length and `snapshot_sha256` with the
digest from the single byte read. Do not compare `source_total_count` with
`record_count`, resolve or access the stored `snapshot_path`, or reject
`metadata`/other extra fields. All malformed or mismatched manifest conditions
raise `SupervisedSnapshotBuildError("hourly-count manifest does not match snapshot")`.

`_observations_from_hourly_count_frame` must construct only `location_id`,
`observed_at`, and `pedestrian_count`; call `melbourne_observed_at` for every
source date/hour pair and use `int` conversion after validation. Do not copy
database-only fields or attempt aggregation.

`_require_calendar_coverage` must normalize each generated target to
`Australia/Melbourne` before calling `holiday_calendar.contains(date_value)`:

```python
target_dates = {
    pd.Timestamp(value).tz_convert("Australia/Melbourne").date()
    for value in supervised["target_observed_at"]
}
if not all(holiday_calendar.contains(value) for value in target_dates):
    raise SupervisedSnapshotBuildError(
        "holiday calendar does not cover generated target dates"
    )
```

For output, reject `os.path.lexists(output_path)` before work, make its parent,
then create a `tempfile.mkstemp` sibling in that same parent and close its
descriptor. Call `supervised.to_csv(temp_path, index=False)` and
`read_supervised_csv(temp_path)`. Publish only with
`os.link(temp_path, output_path)`, which atomically fails with `FileExistsError`
if a competing process created the destination; never use `Path.replace` or
`rename`, because they can overwrite on some platforms. On successful link,
unlink the temporary sibling. Map `FileExistsError` to
`SupervisedSnapshotBuildError(f"supervised CSV output already exists: {output_path}")`;
map other `OSError` or `SupervisedCsvError` after input validation to
`SupervisedSnapshotWriteError(f"could not write supervised CSV: {output_path}")`.
Always unlink the temporary path in `finally` when it still exists.

- [ ] **Step 3: Run the focused core suite and inspect the diff**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/modeling/test_supervised_dataset.py -v
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/modeling/supervised_dataset.py tests/unit/modeling/test_supervised_dataset.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/modeling/supervised_dataset.py tests/unit/modeling/test_supervised_dataset.py
git diff --check
```

Expected: all focused tests pass; formatting and diff checks are clean.

- [ ] **Step 4: Commit the core bridge**

```powershell
git add src/urbanflow/modeling/supervised_dataset.py tests/unit/modeling/test_supervised_dataset.py
git commit -m "feat(modeling): build supervised csv from snapshot"
```

### Task 2: Local CLI and executable wrapper

**Files:**
- Create: `src/urbanflow/modeling/supervised_dataset_cli.py`
- Create: `scripts/build_supervised_csv.py`
- Create: `tests/unit/modeling/test_supervised_dataset_cli.py`

**Interfaces:**
- Consumes: `HolidayCalendar.from_json_file`, `build_supervised_csv_from_hourly_snapshot`, `SupervisedSnapshotBuildError`, and `SupervisedSnapshotWriteError` from Task 1.
- Produces: `build_parser()` and `main(argv: Sequence[str] | None = None) -> int`, with stdout-only success JSON.

- [ ] **Step 1: Write failing CLI and wrapper tests**

Use the Task 1 helpers to write valid snapshot inputs. Assert the parser requires exactly three positional paths and `--holiday-calendar`; no `--horizons`, database, network, artifact, or model option exists. The 200-hour fixture intentionally has one partial final Melbourne date, so its successful validation warning count is `1` (`INCOMPLETE_HOUR_COVERAGE`), not zero.

```python
def test_main_writes_only_json_for_a_valid_local_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot, manifest, calendar = valid_inputs(tmp_path)
    output = tmp_path / "supervised.csv"

    exit_code = main([str(snapshot), str(manifest), str(output), "--holiday-calendar", str(calendar)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "output_path": str(output),
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "source_row_count": 200,
        "supervised_row_count": 4800,
        "training_row_count": 4500,
        "validation_warning_count": 1,
    }
```

Add parametrized input-error tests for a missing snapshot, malformed manifest,
invalid calendar, insufficient calendar coverage, and existing output; each
must yield exit `2`, blank stdout, and `error:` on stderr. Monkeypatch
`build_supervised_csv_from_hourly_snapshot` to raise
`SupervisedSnapshotWriteError` and assert exit `1`. Add a subprocess test that
`scripts/build_supervised_csv.py --help` returns zero and mentions
`--holiday-calendar`.

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/modeling/test_supervised_dataset_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError` for the CLI module and missing script.

- [ ] **Step 2: Implement parser, error mapping, and JSON boundary**

Create the CLI following the existing artifact CLI conventions.

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a supervised CSV from a verified hourly-count snapshot."
    )
    parser.add_argument("snapshot_path", type=Path)
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--holiday-calendar", type=Path, required=True)
    return parser


def _result_summary(result: SupervisedSnapshotBuildResult) -> dict[str, object]:
    return {
        "output_path": str(result.output_path),
        "snapshot_sha256": result.snapshot_sha256,
        "source_row_count": result.source_row_count,
        "supervised_row_count": result.supervised_row_count,
        "training_row_count": result.training_row_count,
        "validation_warning_count": result.validation_warning_count,
    }
```

In `main`, parse the existing `HolidayCalendar` before invoking the builder.
Catch `LightGBMArtifactError` and `SupervisedSnapshotBuildError` together,
printing `error: {exc}` to stderr and returning `2`. Catch
`SupervisedSnapshotWriteError` separately, printing the same prefix and
returning `1`. On success, use `json.dumps(_result_summary(result), sort_keys=True)`
and return `0`.

Create the wrapper exactly as follows:

```python
from urbanflow.modeling.supervised_dataset_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the focused CLI suite**

Run:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/modeling/test_supervised_dataset.py tests/unit/modeling/test_supervised_dataset_cli.py -v
& .\.venv\Scripts\python.exe scripts/build_supervised_csv.py --help
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/modeling/supervised_dataset_cli.py scripts/build_supervised_csv.py tests/unit/modeling/test_supervised_dataset_cli.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/modeling/supervised_dataset_cli.py scripts/build_supervised_csv.py tests/unit/modeling/test_supervised_dataset_cli.py
```

Expected: tests pass, help exits zero, and static checks are clean.

- [ ] **Step 4: Commit the local CLI**

```powershell
git add src/urbanflow/modeling/supervised_dataset_cli.py scripts/build_supervised_csv.py tests/unit/modeling/test_supervised_dataset_cli.py
git commit -m "feat(modeling): add supervised dataset cli"
```

### Task 3: Operator documentation and full verification

**Files:**
- Modify: `README.md:288-301`

**Interfaces:**
- Consumes: the `scripts/build_supervised_csv.py` command delivered by Task 2.
- Produces: a documented, opt-in local bridge whose output is the existing evaluation/artifact input and whose limits are explicit.

- [ ] **Step 1: Add the README bridge section immediately after the modeling-feature overview**

Insert a `### Build a supervised CSV from a validated hourly snapshot` subsection after the existing explanation that modeling is DataFrame-first and before `## Train a local Ridge baseline`.

````markdown
### Build a supervised CSV from a validated hourly snapshot

The model evaluators and local LightGBM artifact exporter consume supervised
rows, not raw City of Melbourne exports. Build those rows only from a local
hourly-count snapshot and its matching ingestion manifest:

```powershell
python scripts/build_supervised_csv.py `
  data/raw/melbourne/hourly_counts/extracted_at=.../records.csv `
  data/manifests/hourly_counts/<timestamp>.json `
  data/processed/modeling/supervised_rows.csv `
  --holiday-calendar data/processed/modeling/holiday_calendar.json
```

The command is local and does not download data, connect to PostgreSQL, train a
model, or create a forecast. It verifies the source manifest, preserves missing
observations as missing feature markers, and rejects duplicate sensor-hour rows
or a calendar that does not cover all 1–24-hour target dates. Generated outputs
remain local and ignored by Git.
````

- [ ] **Step 2: Run the complete quality gate and local command smoke**

Run:

```powershell
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\python.exe scripts/build_supervised_csv.py --help
git diff --check
git status --short --branch
```

Expected: no Ruff issues, formatter reports all files already formatted, all
tests pass, wrapper help exits zero, diff check is clean, and only intended
tracked source/test/documentation changes exist.

- [ ] **Step 3: Commit the documentation**

```powershell
git add README.md
git commit -m "docs: explain snapshot supervised csv workflow"
```

## Plan self-review

- [ ] Every requirement in `2026-08-04-snapshot-supervised-csv-design.md` maps to Task 1, 2, or 3: verified snapshot provenance, same-frame validation, source-field conversion, direct horizons, strict calendar coverage, warning/error policy, atomic output, local CLI, offline tests, and operator documentation.
- [ ] Run the prescribed unfinished-marker scan and ensure it produces no matches; remove any ambiguous instruction rather than leaving an incomplete task.
- [ ] Confirm the public names are identical across tasks: `SupervisedSnapshotBuildError`, `SupervisedSnapshotWriteError`, `SupervisedSnapshotBuildResult`, `build_supervised_csv_from_hourly_snapshot`, `build_parser`, and `main`.
- [ ] Confirm no task adds a database connection, network access, artifact creation, model training, Dashboard change, or configuration migration.
