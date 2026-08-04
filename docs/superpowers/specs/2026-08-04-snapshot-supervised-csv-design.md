# Snapshot-First Supervised CSV Bridge Design

## Context and decision

The approved local real-serving demonstration cannot use the repository's
current checked-in example snapshot: it covers only one day and cannot satisfy
the API's 168-hour forecast-input contract. The existing application already
has PostgreSQL read adapters, an artifact-backed LightGBM provider, and a
Dashboard client. The missing reusable boundary is a trustworthy path from a
validated City of Melbourne hourly-count snapshot to the exact direct
multi-horizon supervised CSV consumed by the existing evaluation and artifact
commands.

The broader 5.1b goal is therefore deliberately decomposed:

1. This slice builds the offline, snapshot-first supervised-CSV bridge.
2. A later operator runbook slice provisions local PostgreSQL, acquires a
   sufficiently long public snapshot, loads it, evaluates and exports an
   artifact, and captures real API/Dashboard evidence.

This separation avoids mixing a new data contract, Windows service
installation, public-network ingestion, database state, model training, and
UI evidence into one unreviewable change.

## Goal

Provide one opt-in local CLI that verifies an immutable hourly-count snapshot
and its provenance manifest, builds the existing leakage-safe direct
`1..24`-hour supervised rows from that same in-memory data, and atomically
writes a new supervised CSV that can be passed unchanged to the established
LightGBM evaluation and artifact-export commands.

## Non-goals

- Downloading public data, connecting to PostgreSQL, applying migrations, or
  installing PostgreSQL.
- Training, evaluating, serializing, or loading a model artifact.
- Changing the existing feature contract, feature horizons, lag behavior,
  missing-value behavior, model selection, API routes, or Dashboard UI.
- Inferring, downloading, or hard-coding a public-holiday calendar.
- Selecting a best sensor, repairing gaps, deduplicating source data, adding
  Parquet/batch processing, or presenting synthetic predictions.

## Architecture

Add a narrowly scoped modeling boundary rather than modifying the feature
builders or database loaders:

```text
hourly-count CSV + source manifest + explicit holiday JSON
  -> read CSV once
  -> verify manifest SHA-256 and record count
  -> validate the same DataFrame
  -> map City of Melbourne fields to canonical observations
  -> build_supervised_frame(horizons=1..24)
  -> atomically write supervised CSV
  -> existing evaluation / existing artifact exporter
```

The new module owns file/provenance checks, snapshot-to-observation conversion,
and safe output writing. Existing `urbanflow.features` continues to own all
feature semantics. Existing artifact code continues to own holiday-calendar
parsing and artifact integrity.

### Files and responsibilities

| File | Responsibility |
| --- | --- |
| `src/urbanflow/modeling/supervised_dataset.py` | Verify snapshot provenance, validate one in-memory snapshot, convert observations, build the supervised frame, and atomically write a new CSV. |
| `src/urbanflow/modeling/supervised_dataset_cli.py` | Parse local paths, load the strict existing holiday calendar, emit JSON on success, and map failures to stable exit codes. |
| `scripts/build_supervised_csv.py` | Thin executable wrapper around the CLI `main`. |
| `tests/unit/modeling/test_supervised_dataset.py` | Pure conversion, provenance, calendar, atomic-write, and timestamp tests. |
| `tests/unit/modeling/test_supervised_dataset_cli.py` | CLI JSON, exit-code, and wrapper tests. |
| `README.md` | Document the explicit local bridge and its relationship to later evaluation and serving steps. |

## Data and provenance contract

The CLI accepts three required local inputs:

```powershell
python scripts/build_supervised_csv.py `
  data/raw/melbourne/hourly_counts/extracted_at=.../records.csv `
  data/manifests/hourly_counts/<timestamp>.json `
  data/processed/modeling/supervised_rows.csv `
  --holiday-calendar data/processed/modeling/holiday_calendar.json
```

The positional arguments are `snapshot_path`, `manifest_path`, and
`output_csv`. `--holiday-calendar` is required. No horizon flag is exposed:
the output is always compatible with the API's direct `1..24`-hour serving
contract.

The source manifest must be schema version `1`, declare
`dataset == "hourly_counts"`, contain non-empty source URL, extraction
timestamp, and stored snapshot-path provenance, and carry non-boolean,
non-negative integer `record_count` and `source_total_count` plus a lowercase
64-hex `snapshot_sha256`. `record_count` and `snapshot_sha256` must match the
exact bytes and rows read from `snapshot_path`; `source_total_count` is source
provenance and is not required to equal the selected snapshot row count. Its
stored snapshot path remains provenance information rather than a
machine-specific path requirement, so a valid ignored snapshot can be replayed
from another local worktree without accessing that old path. A mismatch is a
hard failure; the output file must not be created.

The holiday JSON is parsed by the existing `HolidayCalendar.from_json_file`.
Its explicit coverage must include every local date represented by generated
`target_observed_at` values, including the final 24 target hours. The bridge
passes only `calendar.public_holidays` into `build_supervised_frame`; it does
not invent holiday data.

## Conversion contract

`build_supervised_csv_from_hourly_snapshot(...)` exposes this result:

```python
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

It reads one raw byte snapshot exactly once, parses it with the same
hourly-count CSV semantics (`dtype=str`, `keep_default_na=False`), validates
that same in-memory DataFrame with `validate_hourly_counts_frame`, then maps
only these canonical observation fields:

| Snapshot field | Observation field |
| --- | --- |
| `location_id` | `location_id` |
| `sensing_date` + `hourday` | `observed_at`, via `melbourne_observed_at` |
| `pedestriancount` | `pedestrian_count` |

`observed_at` is therefore timezone-aware in `Australia/Melbourne`, exactly as
the database loader interprets the source. The implementation calls the
existing `build_supervised_frame` with its fixed default horizons and retains
missing count/target rows. It must not interpolate a pedestrian count, use
future data recursively, or rewrite the existing weather-missing feature
contract.

Hard validation errors stop the run. `DUPLICATE_SENSOR_HOUR` is upgraded from a
snapshot warning to a hard bridge error because the feature contract forbids
duplicate `location_id`/timestamp pairs. `DUPLICATE_SOURCE_ID` and
`INCOMPLETE_HOUR_COVERAGE` remain reported warnings: incomplete hours become
explicit missing markers rather than fabricated observations. A later serving
preflight will choose only a sensor that genuinely has 168 consecutive observed
hours.

`training_row_count` is the number of generated rows with a non-missing
target. It is evidence only; the existing model feature selector remains the
authority for trainable feature validation.

## Safe output and CLI behavior

The output CSV must be a new local path. Existing output files are never
overwritten, including when another process creates the destination during a
run. The builder writes to a temporary sibling path, verifies that the CSV can
be read back through `read_supervised_csv` with offset-aware timestamp columns,
then publishes with same-directory non-overwriting `os.link` and removes the
temporary sibling. It must not use `replace` or `rename`, which can overwrite on
some platforms. Any write or publish failure removes the temporary file and
leaves no partial output behind.

On success, the CLI writes one JSON object to stdout with the result fields
needed for operator evidence, including row counts, warning count, snapshot
SHA-256, and output path. It writes nothing else to stdout. Paths may be shown
locally, but no database URL, credential, artifact directory, or remote model
identifier exists in this slice.

| Condition | stderr / exit code |
| --- | --- |
| Success | JSON / `0` |
| Missing/unreadable snapshot or manifest, manifest mismatch, validation error, duplicate sensor-hour, invalid calendar/coverage, or existing output | `error: ...` / `2` |
| A valid conversion that cannot be safely written or atomically published | `error: ...` / `1` |

Stable error wording covers at least these conditions:

```text
hourly-count manifest does not match snapshot
hourly-count snapshot validation failed: <codes>
hourly-count snapshot contains duplicate sensor-hour rows
holiday calendar does not cover generated target dates
supervised CSV output already exists: <path>
could not write supervised CSV: <path>
```

## Tests and acceptance

The new unit suite uses only temporary CSV/JSON fixtures and has no network,
PostgreSQL, MLflow server, or model-artifact dependency. It proves:

1. A valid 200-hour single-sensor snapshot produces exactly `200 * 24` direct
   horizon rows; all generated rows retain the expected calendar and
   all-weather-missing features.
2. The resulting CSV round-trips through `read_supervised_csv`, preserving
   offset-aware timestamps and UTC instants across a Melbourne daylight-saving
   boundary.
3. All eight manifest fields are enforced before output creation, including
   SHA-256, row count, schema/dataset, timestamp, count types, and provenance
   text. A stale stored provenance path is deliberately accepted when the
   supplied snapshot bytes match, proving a snapshot can be replayed from a
   relocated worktree.
4. Schema/direction-total validation failures, empty/unreadable input,
   duplicate sensor-hours, invalid holiday JSON, and insufficient holiday
   coverage fail without creating output.
5. Missing source hours produce explicit missing markers; counts are never
   filled in.
6. Existing output is byte-for-byte unchanged after refusal; simulated write,
   round-trip, or non-overwriting publish failure leaves no temporary artifact.
7. The CLI returns exactly the documented JSON/exit codes, and the script
   wrapper exposes `--help`.

The slice passes the repository's normal Ruff check, formatter check, and full
pytest suite. Default API and Dashboard tests remain fully offline.

## Operator handoff after this slice

The subsequent local-serving demonstration uses an explicit, operator-owned
Windows PostgreSQL installation and a separately acquired public snapshot with
at least five complete months of coverage, allowing the existing three-month
validation evaluation. It will load already-validated snapshots through the
existing database loader, create an explicit local Victorian holiday JSON,
evaluate and export the existing LightGBM bundle, and configure FastAPI with
both local paths.

Historical public data must remain labeled historical. With a historical
cutoff, `/health` is expected to be `degraded` when freshness is unconfigured
or stale; it must not be made `ok` by substituting model-training metadata for
the database cutoff. The Dashboard will continue to request data only through
FastAPI and will show a selected sensor's real returned history and direct
forecast only after that later configured run succeeds.

## Explicit deferrals

This design defers Windows service automation, Docker/Compose, cloud
databases, CI PostgreSQL, public-holiday downloading, weather ingestion,
artifact registries, MLflow serving, retraining schedules, forecast persistence,
maps, cross-sensor aggregates, monitoring/alerts, and any user-facing synthetic
or production-quality claim.
