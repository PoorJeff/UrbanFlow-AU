# Data Validation Design

## Goal

Add the first data-quality boundary for UrbanFlow AU: validate local Melbourne raw
snapshots after ingestion and before database persistence, orchestration, feature
engineering, or modeling.

This slice turns the project requirements' Pandera validation step into a focused,
testable layer. It will validate the two raw datasets already ingested by the project:

- `sensor_locations` JSON snapshots.
- `hourly_counts` CSV snapshots.

The output will be a structured validation report that can be read by developers now
and reused later by Prefect, PostgreSQL loading, and CI smoke checks.

## Source Context

The current ingestion layer writes immutable raw snapshots and manifests:

```text
data/raw/melbourne/sensor_locations/extracted_at=<YYYYMMDDTHHMMSSZ>/records.json
data/raw/melbourne/hourly_counts/extracted_at=<YYYYMMDDTHHMMSSZ>/records.csv
data/manifests/<dataset>/<YYYYMMDDTHHMMSSZ>.json
```

The project requirements name `pandera` as the data-validation tool and recommend
`pandas` as the default DataFrame engine. This validation slice will therefore introduce
`pandas` and `pandera` together, while keeping raw ingestion itself independent from
DataFrames.

One local hourly-count smoke snapshot from 2025-01-01 has already confirmed that
`hourday` values use the 0-23 range and that `direction_1 + direction_2` matched
`pedestriancount` for that sample. The validation rules will still treat source-data
coverage issues as reportable diagnostics rather than silently assuming all future ranges
are complete.

## Considered Approaches

### 1. Build a standalone validation package and report writer (selected)

Add `src/urbanflow/validation/` with dataset-specific validators and a shared report
contract. Validation reads existing snapshots, checks schema and row-level rules, and
writes a JSON report under `reports/data_quality/`.

This keeps the next step small, testable, and reusable. It also creates the boundary that
future database loading and Prefect flows can call without duplicating quality checks.

### 2. Combine validation with PostgreSQL loading

This would move the project closer to persistence, but it couples two failure domains:
data quality and database writes. A validation failure should be understandable without
also debugging SQLAlchemy sessions, migrations, or local PostgreSQL setup.

### 3. Keep validation inside the ingestion parsers

The current ingestion parsers already perform useful source-shape checks. Extending only
those checks would avoid new dependencies, but it would not cover DataFrame-level schema
rules, uniqueness, duplicate keys, row-count diagnostics, or reusable report output.

## Architecture

The implementation will create a focused validation package:

- `src/urbanflow/validation/reports.py`
  - Define `ValidationIssue`, `ValidationMetric`, `ValidationReport`, and JSON
    serialization helpers.
  - Keep the report format independent from Pandera internals so future callers receive
    stable output.
- `src/urbanflow/validation/snapshot_readers.py`
  - Load normalized sensor-location JSON snapshots into pandas DataFrames.
  - Load hourly-count CSV snapshots into pandas DataFrames with explicit dtypes where
    practical.
- `src/urbanflow/validation/sensor_locations.py`
  - Own the Pandera schema and dataset-specific checks for `sensor_locations`.
- `src/urbanflow/validation/hourly_counts.py`
  - Own the Pandera schema and dataset-specific checks for `hourly_counts`.
- `src/urbanflow/validation/pipeline.py`
  - Route a dataset name and snapshot path to the correct validator.
  - Optionally write the report to a caller-provided reports root.
- `src/urbanflow/validation/cli.py`
  - Provide a testable command entry point.
- `scripts/validate_snapshot.py`
  - Thin wrapper around the CLI module.

The package will not import from `urbanflow.ingestion` except for stable dataset constants
where useful. Validation should be able to run against any matching snapshot path, not only
snapshots produced in the same process.

## Validation Rules

### Sensor locations

Hard failures:

- Required columns are present:
  - `location_id`
  - `sensor_description`
  - `sensor_name`
  - `installation_date`
  - `status`
  - `latitude`
  - `longitude`
- `location_id` is integer-like, positive, and unique within the snapshot.
- `sensor_description`, `sensor_name`, and `status` are non-empty strings after trimming.
- `latitude` is numeric and between -90 and 90.
- `longitude` is numeric and between -180 and 180.

Allowed but reported as metrics:

- `installation_date` may be null because the live City of Melbourne source has already
  produced a valid sensor record with a null installation date.
- Status-value distribution is recorded so later slices can decide whether to filter
  inactive sensors.

### Hourly counts

Hard failures:

- Required columns are present in the raw CSV snapshot:
  - `id`
  - `location_id`
  - `sensing_date`
  - `hourday`
  - `direction_1`
  - `direction_2`
  - `pedestriancount`
  - `sensor_name`
  - `location`
- `id` is non-empty and unique within the snapshot.
- `location_id` is integer-like and positive.
- `sensing_date` is parseable as a date.
- `hourday` is integer-like and between 0 and 23.
- `direction_1`, `direction_2`, and `pedestriancount` are non-negative integer-like
  counts.
- When both direction fields and `pedestriancount` are present, `direction_1 +
  direction_2 == pedestriancount`.
- `sensor_name` and `location` are non-empty strings after trimming.

Diagnostics, not first-version hard failures:

- Duplicate `(location_id, sensing_date, hourday)` keys are counted and reported. The
  report will fail only if the raw source `id` is duplicated; duplicate sensor-hour keys
  need source investigation before becoming a hard rule.
- Per-day hour coverage by `location_id` is summarized. Missing hours are warnings for
  now because the source may add, retire, or temporarily disable sensors.
- Date range, row count, number of sensors, and hour distribution are recorded as metrics.

## Report Contract

When report writing is requested, validation will produce one JSON report per run:

```text
reports/data_quality/<dataset>/<validated_at>.json
```

The report fields will be:

- `schema_version`: report contract version, starting at `1`.
- `dataset`: `sensor_locations` or `hourly_counts`.
- `snapshot_path`: path that was validated.
- `validated_at`: UTC timestamp.
- `passed`: boolean.
- `row_count`: number of records inspected.
- `errors`: list of hard-failure issues.
- `warnings`: list of diagnostics that need attention but do not block the first version.
- `metrics`: small structured values such as duplicate counts, date range, sensor count,
  status distribution, and hour distribution.

Each issue will include:

- `code`: stable machine-readable issue code.
- `message`: concise human-readable explanation.
- `column`: optional affected column.
- `rows`: optional sample row indexes, capped to keep reports small.

The CLI will print a compact JSON summary to stdout and write the full report when
`--report-root` is provided. Exit code `0` means the report passed; exit code `1` means
validation completed and found hard failures; exit code `2` means the command itself was
invalid or a snapshot could not be read.

## Data Flow

```text
script/CLI arguments
  -> dataset + snapshot path validation
  -> snapshot reader loads pandas DataFrame
  -> dataset Pandera schema checks column presence, dtypes, nullability, ranges
  -> additional dataset checks compute uniqueness, arithmetic, and coverage diagnostics
  -> ValidationReport object
  -> optional JSON report write under reports/data_quality/
  -> compact CLI JSON summary and exit code
```

## Error Handling

- Missing files, unsupported dataset names, unreadable JSON, unreadable CSV, or empty
  snapshots produce a failed validation report when possible.
- If the CLI arguments are invalid before a dataset can be identified, the command exits
  with code `2` and prints a concise error to stderr.
- Pandera failures are converted into `ValidationIssue` entries rather than exposing
  raw exception text as the public contract.
- Report writing uses no overwrite behavior by default. A pre-existing report path raises
  a clear error instead of replacing a previous validation artifact.

## Testing Strategy

Unit tests stay network-free:

- Report dataclass serialization and issue shape.
- Sensor-location valid snapshot passes.
- Sensor-location failures for duplicate IDs, missing required columns, invalid
  coordinates, blank names, and blank status values.
- Sensor-location null `installation_date` is accepted and counted.
- Hourly-count valid CSV passes.
- Hourly-count failures for missing columns, duplicate `id`, invalid dates, out-of-range
  `hourday`, negative counts, and direction-total mismatch.
- Hourly-count diagnostics for duplicate sensor-hour keys and incomplete hour coverage.
- Pipeline routing for supported and unsupported datasets.
- CLI exit codes for pass, validation failure, and invalid command/read failure.

The full quality gate remains:

```powershell
& .\.venv\Scripts\python.exe -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
```

In an isolated worktree, tests must set `PYTHONPATH=src` so they exercise the worktree
source tree rather than the main checkout's editable install.

## Out of Scope

This slice does not implement:

- PostgreSQL tables or migrations.
- Prefect flows.
- Parquet conversion.
- Top-10 sensor selection.
- Cleaning, imputation, or feature engineering.
- Weather, holiday, or event enrichment.
- Streamlit, Plotly, FastAPI, model training, MLflow, or Evidently.
- CI enforcement of validation against large live snapshots.

Those belong to later planned delivery slices.

## Success Criteria

- A developer can validate a local `sensor_locations` JSON snapshot and receive a
  structured report.
- A developer can validate a local `hourly_counts` CSV snapshot and receive a structured
  report.
- Hard failures block downstream processing through a clear `passed=false` report.
- Warnings and metrics preserve useful data-quality evidence without prematurely rejecting
  source data that may require domain investigation.
- Unit tests cover validation behavior without network access.
- Ruff checks, format checks, and pytest pass before integration.
