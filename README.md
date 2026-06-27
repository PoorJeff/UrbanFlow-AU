# UrbanFlow AU

UrbanFlow AU is an end-to-end platform for forecasting hourly pedestrian demand at selected City of Melbourne sensor locations. It will connect reproducible public-data ingestion, leakage-safe time-series evaluation, model serving, an operations dashboard, and MLOps monitoring.

> **Project status:** foundation stage. The data pipeline and measured model results have not been implemented yet; no performance claims are made.

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

## Planned delivery slices

1. Melbourne sensor and hourly-count ingestion with immutable snapshots and manifests. Sensor-location ingestion is runnable locally; hourly-count ingestion has a bounded CSV export pipeline.
2. Data validation, PostgreSQL persistence, and Prefect orchestration.
3. Leakage-safe features, rolling-origin backtests, and MLflow tracking.
4. FastAPI forecasts, Streamlit operations views, and Evidently monitoring.
5. Docker Compose packaging, evaluation evidence, screenshots, and portfolio documentation.

## Data policy

The repository will contain only small deterministic fixtures, sample data, and manifests. Full raw data, secrets, model artifacts, and local experiment stores remain untracked.
