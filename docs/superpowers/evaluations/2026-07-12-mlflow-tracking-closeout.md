# MLflow tracking closeout evaluation

Date: 2026-07-12

## Decision

Close the first MLflow tracking stage.

Do not add another smoke test or a separate issue note before moving to the next
slice. The current code already has a real file-backed MLflow smoke test, and
the manual local workflow is documented in the README.

## Scope evaluated

This closeout covers the local experiment-tracking path for existing Ridge and
LightGBM evaluation artifacts:

- consume existing evaluation JSON summaries;
- optionally attach existing Markdown reports;
- log params, metrics, tags, and report artifacts through MLflow;
- support both Ridge and LightGBM through the same tracking path;
- keep training, database ingestion, and generated MLflow stores outside the
  tracking command.

## Evidence

| Area | Status | Evidence |
| --- | --- | --- |
| Dependency and adapter seam | Complete | `mlflow>=3,<4`, `MLflowTrackingConfig`, `MLflowRunResult`, fake-adapter unit coverage |
| Tracking implementation | Complete | `track_evaluation_summary(...)` logs params, tags, final metrics, validation-step metrics, JSON summaries, and optional Markdown reports |
| CLI workflow | Complete | `scripts/track_modeling_evaluation.py` supports `ridge`, `lightgbm`, `--tracking-uri`, `--experiment-name`, and repeated `--tag key=value` |
| Expected user errors | Complete | CLI-level validation maps expected failures to exit code `2` |
| Real local file store | Complete | Unit smoke coverage uses a temporary `file://` tracking URI and verifies output stays outside the repository working directory |
| User-facing workflow | Complete | README documents `MLFLOW_ALLOW_FILE_STORE=true`, both tracking commands, and `mlflow ui --backend-store-uri .\mlruns --port 5000` |
| Repository hygiene | Complete | `mlruns/` remains ignored; no run stores, model binaries, data snapshots, or secrets are committed |

## Closeout criteria

| Criterion | Result | Notes |
| --- | --- | --- |
| Tracking command does not retrain models | Pass | It consumes pre-existing summary/report artifacts only |
| Supervised CSV is not logged as an artifact | Pass | The tracking surface logs evaluation evidence, not source data snapshots |
| Ridge and LightGBM share one implementation path | Pass | Model name is validated at the CLI boundary and passed into the same tracking function |
| Metric names are stable and documented by tests | Pass | Unit tests cover flattened final metrics, baseline metrics, relative improvement, and validation-window step metrics |
| Local file-backed MLflow behavior is proven | Pass | Existing smoke test exercises the real MLflow package with a temporary store |
| Manual user path is documented | Pass | README includes commands and the required MLflow file-store environment variable |
| No generated artifacts are committed | Pass | Current design keeps `mlruns/` and future model artifacts local/generated |

## Non-blockers for future slices

These are useful next steps, but they should not block closing this stage:

- model artifact persistence and registry promotion;
- Docker Compose or remote MLflow tracking service;
- FastAPI forecast serving endpoint;
- Streamlit dashboard around forecasts and evaluation evidence;
- Evidently-style monitoring/reporting;
- database-backed training/evaluation reads instead of local CSV inputs.

## Recommendation

Start the next slice with a small design document for FastAPI forecast serving.
Keep serving separate from tracking: serving should load an explicit model
artifact or fixture path, while MLflow remains the audit trail for evaluation
evidence until model registry work is intentionally added.
