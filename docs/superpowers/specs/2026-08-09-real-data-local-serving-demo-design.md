# Real-Data Local Serving Demo Design

## Goal

Produce one honest, reproducible local UrbanFlow AU demonstration using City of
Melbourne data for `location_id=1` from 2025-01-01 through 2025-05-31. The demo
must connect the existing ingestion, validation, modeling, PostgreSQL, FastAPI,
and Streamlit boundaries without synthetic forecasts or production claims.

The selected sensor is `Bourke Street Mall (North)`. A read-only source query
on 2026-08-09 returned 3,624 rows for the bounded interval, exactly 151 days
times 24 hours. Project validation remains authoritative for duplicates,
missing sensor-hours, and the 168-hour serving window.

## Decisions

- Fast-forward the already-verified location-scoped ingestion work into `main`
  before beginning the demo branch.
- Reuse the existing `codex/local-serving-demo` worktree.
- Add `location_id` only to the existing Prefect ingestion-flow boundary. The
  value flows unchanged to `ingest_hourly_counts`; no generic source query or
  automatic sensor selector is introduced.
- Use the official 2025 Victorian public-holiday list published by Business
  Victoria and cross-referenced by the Fair Work Ombudsman. The local calendar
  covers 2025-01-01 through 2025-06-01 so the last historical cutoff can serve
  a 24-hour forecast.
- Use one local PostgreSQL instance. Starting the installed Docker Desktop and
  a disposable PostgreSQL container is an execution prerequisite, not Docker
  Compose product packaging.
- Keep raw snapshots, processed CSVs, PostgreSQL state, MLflow state, and model
  bundles untracked. Commit only small evidence summaries, reports, screenshots,
  and reproducibility documentation.

## Data and model flow

```text
Melbourne Open Data
  -> scoped Prefect ingestion (location_id=1, five months)
  -> immutable snapshots + manifests
  -> validation reports + PostgreSQL rows
  -> manifest-verified supervised CSV
  -> equal-window Ridge and LightGBM rolling-origin evaluation
  -> Seasonal Naive comparisons in both summaries
  -> final-fit LightGBM artifact
  -> configured FastAPI
  -> Streamlit Today / Explore / Forecast
```

The evaluation reports record the observed MAE, RMSE, and WAPE. LightGBM is not
required to outperform either comparison model. The serving artifact always
records the supervised CSV hash, evaluation-summary reference, training cutoff,
holiday coverage, and generated model version.

## Public interface change

`scripts/run_ingestion_flow.py` gains optional `--location-id POSITIVE_INTEGER`.
The flow passes that value through `run_ingestion_flow` and
`ingest_hourly_counts_task` to the existing scoped ingestion pipeline. Omitting
the option preserves the current all-sensor behavior and manifest contract.

The CLI keeps its existing exit semantics: malformed identifiers are argparse
usage errors, configuration errors return `2`, and source, validation, or
storage failures return `1`.

## Runtime behavior

The configured API uses the historical database and trusted local artifact.
Because the demo data is historical, `GET /health` may truthfully return
`degraded` with HTTP 200; no freshness threshold is set merely to force `ok`.
The sensor list, history, forecast, and model-metrics routes must return data
from the real pipeline. The Dashboard must use its existing HTTP client and may
not load a sample or fallback forecast.

## Evidence and acceptance

- The hourly manifest records `location_id=1`, the exact date-and-location
  source predicate, 3,624 rows if the source remains unchanged, and its SHA256.
- Validation passes and the latest database history has at least 168 contiguous
  hourly observations.
- Ridge and LightGBM use the same three validation months and final test month;
  both retain their Seasonal Naive comparisons.
- FastAPI returns successful sensor, bounded history, 24-hour forecast, and
  real metrics responses against the configured database and artifact.
- Streamlit visibly renders real results on Today, Explore, and Forecast, with
  one screenshot per page.
- README and a compact runbook identify exact commands, source provenance,
  limitations, and untracked outputs.
- Ruff, formatter check, the full pytest suite, configured serving smoke, and
  repository hygiene checks pass before completion.

## Deferred

Evidently, a Monitoring page, Docker Compose packaging, online deployment,
weather features, model registry, automated retraining, multi-sensor expansion,
and production forecasting claims remain separate slices.
