# UrbanFlow AU

UrbanFlow AU is an end-to-end platform for forecasting hourly pedestrian demand at selected City of Melbourne sensor locations. It will connect reproducible public-data ingestion, leakage-safe time-series evaluation, model serving, an operations dashboard, and MLOps monitoring.

> **Project status:** foundation and local-baseline stage. Local ingestion,
> persistence, feature-building, and baseline evaluation slices are in place,
> but no production forecasting performance claims are made.

## Requirements

- Python 3.11 (CI reference version)
- Git

The complete product scope is documented in [urbanflow-au_requirements.md](urbanflow-au_requirements.md). The foundation design is in [docs/superpowers/specs/2026-06-20-project-foundation-design.md](docs/superpowers/specs/2026-06-20-project-foundation-design.md).

The project development workflow is documented in [docs/development_workflow.md](docs/development_workflow.md).

## Local development

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the same quality checks used by CI:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

## Run sensor-location ingestion locally

```powershell
python scripts/ingest_sensor_locations.py
```

The command fetches the current City of Melbourne sensor-location dataset and
prints a JSON summary. By default it writes an immutable snapshot below
`data/raw/` and a matching manifest below `data/manifests/`; both are ignored
by Git. Use `--raw-root`, `--manifest-root`, or `--page-limit` to override the
defaults.

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

## Build leakage-safe modeling features

The first modeling foundation is intentionally local and deterministic. It
builds supervised `forecast_horizon=1..24` rows from hourly pedestrian
observations, adds calendar, lag, rolling, missing-marker, and optional weather
columns, and evaluates a Seasonal Naive baseline through chronological split
utilities.

The feature implementation is DataFrame-first so it can be tested without
PostgreSQL, network access, MLflow, or model artifact persistence. The local
Ridge and LightGBM baselines build on the same feature and split contracts;
future slices can add database-backed training reads and MLflow tracking on top
of this path.

## Train a local Ridge baseline

The first trainable model slice fits a leakage-safe Ridge Regression baseline on
the supervised feature rows. It uses the same rolling-origin windows and metrics
as the Seasonal Naive baseline, keeps predictions in DataFrames, and remains
local and deterministic.

This Ridge slice does not add MLflow tracking, database-backed training reads,
or model artifact persistence. Those pieces build on the current local training
and evaluation contract.

To evaluate Ridge from an already-built supervised feature CSV, run:

```powershell
python scripts/evaluate_ridge_baseline.py data/modeling/supervised_rows.csv --validation-months 3
```

The command expects supervised feature rows, not raw City of Melbourne
hourly-count data. It prints a JSON summary with rolling-origin validation and
final-test metrics.

To render the JSON summary into a Markdown report, run:

```powershell
python scripts/evaluate_ridge_baseline.py data/modeling/supervised_rows.csv --validation-months 3 > reports/modeling/ridge_evaluation.json
python scripts/render_ridge_evaluation_report.py reports/modeling/ridge_evaluation.json --output reports/modeling/ridge_evaluation.md
```

The `reports/` directory is for local generated artifacts and is not required
for unit tests.

A checked-in synthetic example report is available at
[`docs/examples/modeling/ridge_evaluation_report.md`](docs/examples/modeling/ridge_evaluation_report.md).

The generated Markdown report includes exact metric tables plus Mermaid
comparison charts for viewers that support Mermaid, such as GitHub. If a viewer
does not render Mermaid charts, the tables remain the source of exact values.
The same report also includes a Ridge versus Seasonal Naive comparison table so
the trainable baseline can be interpreted against a one-week-prior baseline.

## Train a local LightGBM baseline

The LightGBM baseline is the first non-linear trainable model in the project.
It consumes the same supervised feature CSV as Ridge, uses the same
rolling-origin validation and final-test windows, and reports LightGBM metrics
beside Seasonal Naive comparison metrics.

To evaluate LightGBM from an already-built supervised feature CSV, run:

```powershell
python scripts/evaluate_lightgbm_baseline.py data/modeling/supervised_rows.csv --validation-months 3
```

The command prints a JSON summary. The scores are local evaluation results from
the supplied supervised CSV; they are not production performance claims and do
not imply deployed model behavior.

To render the JSON summary into a Markdown report, run:

```powershell
python scripts/evaluate_lightgbm_baseline.py data/modeling/supervised_rows.csv --validation-months 3 > reports/modeling/lightgbm_evaluation.json
python scripts/render_lightgbm_evaluation_report.py reports/modeling/lightgbm_evaluation.json --output reports/modeling/lightgbm_evaluation.md
```

A checked-in synthetic example report is available at
[`docs/examples/modeling/lightgbm_evaluation_report.md`](docs/examples/modeling/lightgbm_evaluation_report.md).

The generated LightGBM report includes exact metric tables, Mermaid metric
charts, and a LightGBM versus Seasonal Naive comparison table. MLflow tracking,
model artifact persistence, feature-importance plots, and production serving
remain out of scope for this local baseline slice.

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

## Load validated snapshots into PostgreSQL

Set a SQLAlchemy-compatible PostgreSQL URL, run migrations, then load validated snapshots:

```powershell
$env:URBANFLOW_DATABASE_URL = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"
alembic upgrade head

$sensorSnapshot = Get-ChildItem data/raw/melbourne/sensor_locations -Filter records.json -Recurse | Select-Object -First 1
python scripts/load_snapshot_to_db.py sensor_locations $sensorSnapshot.FullName

$hourlySnapshot = Get-ChildItem data/raw/melbourne/hourly_counts -Filter records.csv -Recurse | Select-Object -First 1
python scripts/load_snapshot_to_db.py hourly_counts $hourlySnapshot.FullName
```

The database loader validates each snapshot before writing. Validation hard errors stop
the load; validation warnings are reported but do not block insertion.

### Run a local PostgreSQL smoke test

The persistence stage also includes an explicit smoke test for a real local
PostgreSQL database. It creates a temporary schema, writes one synthetic sensor
row and one hourly-count row, verifies the row counts, then drops the schema.

```powershell
$env:URBANFLOW_SMOKE_DATABASE_URL = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"
python scripts/smoke_test_postgres_persistence.py
```

This smoke test is manual by design, so routine unit tests do not require a
running PostgreSQL service.

## Planned delivery slices

1. Melbourne sensor and hourly-count ingestion with immutable snapshots and manifests. Sensor-location ingestion is runnable locally; hourly-count ingestion has a bounded CSV export pipeline.
2. Data validation, PostgreSQL persistence, and Prefect orchestration.
3. Leakage-safe features, rolling-origin backtests, and MLflow tracking.
4. FastAPI forecasts, Streamlit operations views, and Evidently monitoring.
5. Docker Compose packaging, evaluation evidence, screenshots, and portfolio documentation.

## Data policy

The repository will contain only small deterministic fixtures, sample data, and manifests. Full raw data, secrets, model artifacts, and local experiment stores remain untracked.
