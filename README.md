# UrbanFlow AU

UrbanFlow AU is an end-to-end platform for forecasting hourly pedestrian demand at selected City of Melbourne sensor locations. It will connect reproducible public-data ingestion, leakage-safe time-series evaluation, model serving, an operations dashboard, and MLOps monitoring.

> **Project status:** foundation, local-baseline, first FastAPI serving
> boundary, and initial Streamlit operations views. Local ingestion,
> persistence, feature-building, baseline evaluation, reporting, MLflow
> tracking, typed API reads, and guided `Today`, `Explore`, and `Forecast`
> views are in place. PostgreSQL-backed reads and trusted local LightGBM
> artifact forecasts are opt-in; Evidently monitoring, deployment/packaging,
> and production forecasting performance claims are not in place.

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

## Run the FastAPI API boundary

Start the local API from an activated virtual environment:

```powershell
python -m uvicorn urbanflow.api.app:app --reload
```

The unversioned health probe is available at
`http://127.0.0.1:8000/health`. FastAPI also generates interactive OpenAPI
documentation at `http://127.0.0.1:8000/docs` and the raw schema at
`http://127.0.0.1:8000/openapi.json`.

### Health readiness

`GET /health` reports the running API process together with the configured
data store, data freshness, and model provider. Its HTTP result is a runtime
readiness signal:

| Situation | Component result | Overall result |
| --- | --- | --- |
| No database URL | data store, freshness, and model provider are `unconfigured` | `degraded` (200) |
| Readable database, age threshold unset | data store is `available`; freshness is `unconfigured`; model provider reflects artifact configuration | `degraded` (200) |
| Readable, fresh database and valid artifact | all four components are `available` | `ok` (200) |
| Stale, empty, or future-dated database observation | data store is `available`; freshness is `unavailable`; model provider reflects artifact configuration | `degraded` (200) |
| Unreadable database | data store and freshness are `unavailable`; model provider reflects artifact configuration | `unavailable` (503) |
| Invalid configured artifact | model provider is `unavailable`; data-store and freshness results still reflect the database | `degraded` (200), unless the database is unreadable |

Set `URBANFLOW_API_MAX_DATA_AGE_HOURS` to require the latest database
observation to be no older than that many elapsed UTC hours. It is optional:
when unset or blank, freshness is not configured. When set, it must be a
canonical positive integer such as `24`; zero, negative, non-integer, or other
malformed values make app construction fail with a configuration error.

`data_cutoff_at` is the latest observation timestamp actually read from the
database; it is never a model-training cutoff. Even `health.status == "ok"`
does not guarantee that every sensor has the 168 contiguous input rows (or
other request-specific conditions) required to generate a forecast.

Business routes are versioned under `/api/v1`:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/sensors?active_only=true"
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/sensors/101/history?start=2026-07-01T00:00:00Z&end=2026-07-02T00:00:00Z"
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/sensors/101/forecast?horizon=24"
```

This is an honest local API serving boundary, not production forecasting. The
default process does not connect to PostgreSQL or Melbourne Open Data, so the
sensor catalog is empty and default history requests return
`404 sensor_not_found`. A missing or blank `URBANFLOW_DATABASE_URL` also means
that no SQLAlchemy Engine is created and no model artifact is read.

Set `URBANFLOW_DATABASE_URL` explicitly to serve persisted sensor and history
rows through the PostgreSQL read adapter. The database must already have the
project's migrations and persisted data.

For example, start Uvicorn in one PowerShell window:

```powershell
$env:URBANFLOW_DATABASE_URL = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"
python -m uvicorn urbanflow.api.app:app --reload
```

With only the database configured, sensor and history reads work, but forecast
requests return `503 model_unavailable`. In another PowerShell window, read
active sensors and a bounded history range for one returned sensor:

```powershell
$activeSensors = Invoke-RestMethod "http://127.0.0.1:8000/api/v1/sensors?active_only=true"
$locationId = $activeSensors.data[0].location_id
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/sensors/$locationId/history?start=2026-07-01T00:00:00Z&end=2026-07-02T00:00:00Z"
```

Artifact-backed forecasts exist only when both `URBANFLOW_DATABASE_URL` and a
valid, operator-controlled local `URBANFLOW_API_MODEL_ARTIFACT_PATH` are
configured. Export a final-fit LightGBM artifact from an explicit supervised
CSV and holiday calendar:

```powershell
python scripts/export_lightgbm_artifact.py data/modeling/supervised_rows.csv models/lightgbm/local-demo --holiday-calendar data/modeling/holiday_calendar.json
```

`models/` is ignored by Git. The destination contains exactly `model.joblib`
and `manifest.json`; treat the complete directory as trusted local operator
input. It is not an MLflow Registry artifact, and the API never downloads or
registers it remotely.

The holiday calendar must be a JSON object with exactly this shape:

```json
{
  "coverage_start": "2025-01-01",
  "coverage_end": "2026-12-31",
  "public_holidays": ["2025-01-27", "2026-01-26"]
}
```

Dates use ISO `YYYY-MM-DD`; coverage is inclusive, and `public_holidays` must be
sorted, unique, and inside that range. Every requested forecast target date
must be covered. A request outside coverage returns
`503 forecast_unavailable`. Because this first artifact slice has no serving
weather source, export rejects eligible training rows containing observed
`temperature`, `rainfall`, or `wind_speed` values (or weather missing markers
that are not true); serving uses the established all-weather-missing feature
contract.

To serve forecasts, set both variables in the same PowerShell process and start
Uvicorn:

```powershell
$env:URBANFLOW_DATABASE_URL = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"
$env:URBANFLOW_API_MODEL_ARTIFACT_PATH = ".\models\lightgbm\local-demo"
python -m uvicorn urbanflow.api.app:app --reload
```

Then request a direct 1–24 hour forecast from another PowerShell window:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/v1/sensors/101/forecast?horizon=24"
```

The safe configuration matrix is intentional: without a database URL the app
creates no Engine and does not read an artifact; with a database alone, a
missing artifact, or an invalid artifact, sensor/history reads remain available
but forecast returns `503 model_unavailable`. Invalid current history or
holiday-calendar coverage returns `503 forecast_unavailable`, and storage
failure returns `503 data_store_unavailable`. No configuration fabricates a
prediction.

`scripts/smoke_test_lightgbm_forecast.py` is an opt-in integration check for a
disposable PostgreSQL schema and temporary local artifact. It is not part of
routine tests and requires an intentionally configured
`URBANFLOW_SMOKE_DATABASE_URL`.

### Run the configured serving smoke manually

The configured-serving smoke requires an explicit database URL and never falls
back to `URBANFLOW_DATABASE_URL`:

```powershell
$env:URBANFLOW_SMOKE_DATABASE_URL = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"
python scripts/smoke_test_serving_e2e.py
```

It creates and removes an isolated schema and a temporary local artifact,
starts only a temporary loopback Uvicorn process, and makes no external network
request; all smoke HTTP stays on loopback. It does not start Streamlit, does not
run in CI, and cleans up its temporary resources when the command finishes.

This first serving slice introduces no model registry, retraining, Dashboard,
monitoring, Docker packaging, or production-performance claim.

The model-metrics route reads an existing local evaluation-summary JSON only
when it is explicitly configured. For example, the checked-in summary below is
synthetic evaluation evidence, not a production-model claim:

```powershell
$env:URBANFLOW_API_METRICS_PATH = ".\docs\examples\modeling\lightgbm_evaluation_summary.json"
Invoke-RestMethod http://127.0.0.1:8000/api/v1/model/metrics
```

Without that environment variable, or if the summary is unusable,
`GET /api/v1/model/metrics` returns `503 metrics_unavailable`. When a source
summary lacks `model_version`, `mlflow_run_id`, `mlflow_tracking_uri`, or
`report_path`, the response uses JSON `null`; the API never invents that
metadata and never queries an MLflow server.

## Run the Streamlit operations dashboard

Run the API and dashboard as two local processes. In the first PowerShell
terminal, start and configure the existing FastAPI service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn urbanflow.api.app:app --reload
```

In the second PowerShell terminal, start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The dashboard uses `http://127.0.0.1:8000` by default. Set
`URBANFLOW_DASHBOARD_API_BASE_URL` only when the API runs on another origin:

```powershell
$env:URBANFLOW_DASHBOARD_API_BASE_URL = "http://127.0.0.1:9000"
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The default unconfigured FastAPI process has no persisted sensor data or model
artifact, so the dashboard honestly shows degraded or empty states rather than
inventing results. A user must choose a returned active location before
detailed data loads. The guided workflow starts at `Today`, then links to
`Explore` for bounded returned history or `Forecast` for a direct forecast
request. A real forecast requires the existing API database and model
configuration described above. These views are an initial non-production
operations boundary; monitoring and deployment are still future work.

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

The command above downloads an unfiltered, year-wide City of Melbourne
hourly-count CSV export and prints a JSON summary. For a bounded historical
export from exactly one sensor, provide a date range and its location ID:

```powershell
python scripts/ingest_hourly_counts.py --start-date 2025-01-01 --end-date 2025-05-31 --location-id 101
```

`--location-id` restricts the source query to exactly one sensor, while
`--start-date` and `--end-date` bound the historical window. A successful
export proves only snapshot acquisition and provenance; it does not prove that
the snapshot contains 168 contiguous hourly observations or that it is valid
for a forecast.

Use `--year YYYY` for a full calendar year, or provide both
`--start-date YYYY-MM-DD` and `--end-date YYYY-MM-DD` for a smaller range.
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

The feature and evaluation implementations are DataFrame-first so they can be
tested without PostgreSQL or network access. The local Ridge and LightGBM
baselines build on the same feature and split contracts. A separate local
exporter can fit and persist the trusted LightGBM serving bundle described
above; database-backed training reads remain outside this slice.

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
under `data/processed/` are ignored by Git.

## Train a local Ridge baseline

The first trainable model slice fits a leakage-safe Ridge Regression baseline on
the supervised feature rows. It uses the same rolling-origin windows and metrics
as the Seasonal Naive baseline, keeps predictions in DataFrames, and remains
local and deterministic.

The Ridge evaluator stays local and file-based: it does not read training data
from PostgreSQL or persist fitted model artifacts. MLflow tracking is available
as a separate explicit command that logs existing evaluation evidence.

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
charts, and a LightGBM versus Seasonal Naive comparison table. Feature-
importance plots remain future work. The separate trusted local artifact
exporter and opt-in provider do not turn these evaluation results into a
production-serving claim.

## Track local evaluation artifacts with MLflow

After generating a Ridge or LightGBM JSON summary and optional Markdown report,
log those existing artifacts to MLflow with the explicit tracking command:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
python scripts/track_modeling_evaluation.py lightgbm reports/modeling/lightgbm_evaluation.json --report reports/modeling/lightgbm_evaluation.md
python scripts/track_modeling_evaluation.py ridge reports/modeling/ridge_evaluation.json --report reports/modeling/ridge_evaluation.md
mlflow ui --backend-store-uri .\mlruns --port 5000
```

`MLFLOW_ALLOW_FILE_STORE=true` is required for the local filesystem-backed
`mlruns/` store with current MLflow versions. `mlruns/` is generated local
output and is ignored by Git. If you use a database or remote tracking server,
pass `--tracking-uri` instead of relying on the local default.

The tracking command does not train models, read PostgreSQL, or log the full
supervised CSV. It records local evaluation evidence: run tags, parameters,
final-test metrics, validation-window metrics with MLflow steps, the JSON
summary under `evaluation/`, and the optional Markdown report under `reports/`.
These runs document local baseline evidence, not production performance claims.
Model Registry workflows, Docker Compose MLflow services, and logging model
artifacts into MLflow remain future slices. Those capabilities are distinct
from the trusted local `models/` bundle used by the opt-in forecast provider.

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
python scripts/smoke_test_postgres_api.py
```

The persistence smoke writes and counts synthetic rows. The API smoke creates
and drops its own isolated schema, then checks the PostgreSQL sensor/history
read adapter against exact active/inactive fixture statuses. Both are manual by
design, so routine unit tests do not require a running PostgreSQL service.

## Planned delivery slices

1. Melbourne sensor and hourly-count ingestion with immutable snapshots and manifests. Sensor-location ingestion is runnable locally; hourly-count ingestion has a bounded CSV export pipeline.
2. Data validation, PostgreSQL persistence, and Prefect orchestration.
3. Leakage-safe features, rolling-origin backtests, and MLflow tracking.
4. First FastAPI serving boundary: health, opt-in PostgreSQL-backed
   sensor/history reads, a trusted local artifact-backed direct-forecast
   provider, and local evaluation-summary metrics. The slice remains explicitly
   non-production.
5. Initial Streamlit `Today`, `Explore`, and `Forecast` operations views are
   delivered; Evidently monitoring remains future work.
6. Docker Compose deployment/packaging, evaluation evidence, screenshots, and
   portfolio documentation remain future work.

## Data policy

The repository will contain only small deterministic fixtures, sample data, and manifests. Full raw data, secrets, model artifacts, and local experiment stores remain untracked.
