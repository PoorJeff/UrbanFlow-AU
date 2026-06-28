# PostgreSQL Persistence Design

## Goal

Add the first persistence boundary for UrbanFlow AU: load validated local snapshots into
PostgreSQL tables that downstream feature engineering, API routes, and dashboards can
query consistently.

This slice will implement only the two database tables required by the data already
ingested and validated:

- `sensor_dim`
- `pedestrian_hourly_fact`

It will not introduce Prefect, Docker Compose, API routes, weather data, feature
engineering, modeling, dashboards, or monitoring.

## Source Context

The project already has:

- Sensor-location ingestion to immutable JSON snapshots.
- Hourly-count ingestion to bounded immutable CSV snapshots.
- Pandera-backed validation for both snapshot types.
- CLI validation summaries and optional data-quality reports.

The requirements specify PostgreSQL, SQLAlchemy, and Alembic for database persistence, and
state that database migrations must be managed by Alembic rather than hand-written table
creation only.

Official documentation checked on 2026-06-28:

- SQLAlchemy 2.x ORM documentation for typed declarative mapping with `Mapped` and
  `mapped_column`: <https://docs.sqlalchemy.org/en/20/orm/quickstart.html>
- Alembic tutorial for migration environment layout and revision-based migrations:
  <https://alembic.sqlalchemy.org/en/latest/tutorial.html>
- Psycopg 3 installation documentation for the current PostgreSQL adapter package:
  <https://www.psycopg.org/psycopg3/docs/basic/install.html>

## Considered Approaches

### 1. SQLAlchemy models, Alembic migration, and snapshot loaders (selected)

Create a `src/urbanflow/database/` package with SQLAlchemy 2.x typed models, an Alembic
environment, repository/load functions, and a small CLI for loading validated snapshots.
Loaders call the validation layer before writing. Validation hard failures stop the load;
warnings are allowed and remain visible through the validation report.

This is the smallest useful persistence slice. It creates real database structure and
reusable write paths without coupling the work to orchestration or application serving.

### 2. Store raw snapshots as JSON/CSV blobs first

This would be quick and preserves source files, but the project already keeps immutable
raw snapshots on disk. Putting those blobs into PostgreSQL would not provide the relational
tables needed for features, forecasts, or API queries.

### 3. Build Docker Compose, PostgreSQL, Prefect, and database loading together

This is closer to the final platform, but it mixes infrastructure, orchestration, and data
modeling into one broad change. Failures would be harder to diagnose, and it would slow
the feedback loop before the table contracts are stable.

## Architecture

The implementation will create a focused database package:

- `src/urbanflow/database/__init__.py`
  - Export the public database helpers.
- `src/urbanflow/database/config.py`
  - Read database URL configuration from `URBANFLOW_DATABASE_URL`.
  - Provide README examples for a local development URL, but do not silently invent a
    database URL when neither CLI arguments nor environment variables are set.
- `src/urbanflow/database/engine.py`
  - Build SQLAlchemy engines and session factories.
- `src/urbanflow/database/models.py`
  - Define `Base`, `SensorDim`, and `PedestrianHourlyFact` using SQLAlchemy 2.x typed
    declarative mapping.
- `src/urbanflow/database/time.py`
  - Convert hourly source fields into timezone-aware `observed_at` values.
- `src/urbanflow/database/loaders.py`
  - Load validated sensor-location and hourly-count snapshots.
  - Convert validated snapshot rows into database row dictionaries.
  - Write rows through repository functions inside caller-owned transactions.
- `src/urbanflow/database/repositories.py`
  - Own insert/upsert SQL construction.
  - Keep PostgreSQL conflict behavior isolated from parsing and validation.
- `src/urbanflow/database/cli.py`
  - Provide a small manual loader command.
- `scripts/load_snapshot_to_db.py`
  - Thin wrapper around the CLI module.
- `alembic.ini`
  - Alembic CLI configuration.
- `migrations/env.py`
  - Import database metadata for migration generation and execution.
- `migrations/versions/20260628_0001_create_core_tables.py`
  - Create the first two persistence tables.

## Table Design

### `sensor_dim`

Columns:

- `location_id`: integer primary key.
- `sensor_name`: non-empty text.
- `sensor_description`: non-empty text.
- `latitude`: double precision, non-null.
- `longitude`: double precision, non-null.
- `installation_date`: date, nullable because the live source can emit null values.
- `status`: non-empty text.
- `updated_at`: timezone-aware timestamp, set on insert/update by the database layer.

Constraints:

- Primary key on `location_id`.
- Latitude range check between -90 and 90.
- Longitude range check between -180 and 180.

Load behavior:

- Upsert by `location_id`.
- Update descriptive fields and `updated_at` when an existing sensor appears again.

### `pedestrian_hourly_fact`

Columns:

- `location_id`: integer foreign key to `sensor_dim.location_id`.
- `observed_at`: timezone-aware hourly timestamp.
- `source_sensing_date`: source local calendar date.
- `source_hourday`: source hour bucket, 0-23.
- `pedestrian_count`: non-negative integer.
- `direction_1_count`: non-negative integer.
- `direction_2_count`: non-negative integer.
- `ingested_at`: timezone-aware timestamp set by the database layer.
- `source_snapshot_path`: text path to the local snapshot used for this load.

Constraints:

- Composite primary key on `(location_id, observed_at)`.
- Foreign key from `location_id` to `sensor_dim.location_id`.
- Check constraints for non-negative count columns.
- Check constraint for `source_hourday` between 0 and 23.

Load behavior:

- Upsert by `(location_id, observed_at)`.
- Replace count fields and provenance fields if a later load has the same sensor-hour.
- Require sensors to exist before hourly rows are loaded. The first implementation will
  fail clearly on missing sensor foreign keys instead of auto-creating partial sensors.

## Time Handling

The source provides `sensing_date` plus `hourday`. The database layer will interpret that
pair as a local Melbourne hour using `zoneinfo.ZoneInfo("Australia/Melbourne")`, then
store a timezone-aware `observed_at` in PostgreSQL.

The loader will preserve `source_sensing_date` and `source_hourday` alongside
`observed_at`. This keeps the conversion auditable and gives future data-quality checks a
way to investigate daylight-saving edge cases without reverse-engineering timestamps.

## Data Flow

```text
validated local snapshot path
  -> validate_snapshot(dataset, path)
  -> fail before writes when report.passed is false
  -> read snapshot into DataFrame
  -> transform rows into database row dictionaries
  -> repository upsert statements
  -> SQLAlchemy session transaction
  -> PostgreSQL tables
```

The CLI will accept:

```powershell
$databaseUrl = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"
$sensorSnapshot = Get-ChildItem data/raw/melbourne/sensor_locations -Filter records.json -Recurse | Select-Object -First 1
python scripts/load_snapshot_to_db.py sensor_locations $sensorSnapshot.FullName --database-url $databaseUrl

$hourlySnapshot = Get-ChildItem data/raw/melbourne/hourly_counts -Filter records.csv -Recurse | Select-Object -First 1
python scripts/load_snapshot_to_db.py hourly_counts $hourlySnapshot.FullName --database-url $databaseUrl
```

If `--database-url` is omitted, the CLI will read `URBANFLOW_DATABASE_URL`.

## Error Handling

- Unsupported dataset names fail before validation or database connection.
- Validation hard failures stop before opening a write transaction.
- Snapshot read errors reuse the validation layer's structured failure behavior.
- Missing database URL produces a concise CLI error.
- Database write failures roll back the transaction and propagate a domain-specific
  database load error.
- Hourly rows whose `location_id` is absent from `sensor_dim` fail through the foreign key
  constraint; the loader reports this as an operational loading error.

## Testing Strategy

Automated tests stay network-free and do not require a running PostgreSQL server by
default:

- Model metadata tests verify table names, columns, primary keys, foreign keys, and check
  constraints.
- Transformation tests verify:
  - sensor JSON rows become `sensor_dim` row dictionaries;
  - hourly CSV rows become `pedestrian_hourly_fact` row dictionaries;
  - `observed_at` is timezone-aware;
  - source date/hour provenance is preserved.
- Repository tests compile PostgreSQL insert/upsert statements with SQLAlchemy's
  PostgreSQL dialect to confirm conflict targets.
- Loader tests use fake sessions/repositories where possible to prove validation gating
  and transaction behavior without a live database.
- CLI tests cover argument parsing, missing database URL, successful delegated load, and
  validation-failure exit code.

Optional integration tests may be added later under `tests/integration/database/` and
guarded by `URBANFLOW_TEST_DATABASE_URL`. They should not run in the default CI path until
the project has Docker Compose or another reliable PostgreSQL service.

The standard quality gate remains:

```powershell
& .\.venv\Scripts\python.exe -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
```

In an isolated worktree, tests must set `PYTHONPATH=src` so they exercise the worktree
source tree rather than the main checkout's editable install.

## Out of Scope

This slice does not implement:

- Docker Compose PostgreSQL service.
- Prefect flows.
- Weather, holiday, or event tables.
- Forecast storage.
- Feature engineering tables or model training datasets.
- API routes.
- Streamlit or Plotly.
- MLflow or Evidently.
- Full historical backfill automation.

Those belong to later delivery slices after the first persistence contract is stable.

## Success Criteria

- Alembic can create the `sensor_dim` and `pedestrian_hourly_fact` tables.
- SQLAlchemy models represent the required table contracts.
- A developer can load a validated sensor-location snapshot into `sensor_dim`.
- A developer can load a validated hourly-count snapshot into `pedestrian_hourly_fact`.
- Loading refuses snapshots whose validation report has hard errors.
- Re-loading the same snapshot is idempotent through upsert behavior.
- Unit tests cover model contracts, row transformation, validation gating, and CLI behavior
  without requiring a live PostgreSQL service.
- Ruff checks, format checks, and pytest pass before integration.
