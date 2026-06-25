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

## Planned delivery slices

1. Melbourne sensor and hourly-count ingestion with immutable snapshots and manifests. Sensor-location ingestion now has a tested Python function entry point.
2. Data validation, PostgreSQL persistence, and Prefect orchestration.
3. Leakage-safe features, rolling-origin backtests, and MLflow tracking.
4. FastAPI forecasts, Streamlit operations views, and Evidently monitoring.
5. Docker Compose packaging, evaluation evidence, screenshots, and portfolio documentation.

## Data policy

The repository will contain only small deterministic fixtures, sample data, and manifests. Full raw data, secrets, model artifacts, and local experiment stores remain untracked.
