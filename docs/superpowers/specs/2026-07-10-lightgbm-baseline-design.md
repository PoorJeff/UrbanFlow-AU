# LightGBM Baseline Design

Date: 2026-07-10

## Goal

Add the project's first non-linear trainable model: a local LightGBM global
baseline for pedestrian-count forecasting.

The model should reuse the supervised feature rows, chronological
rolling-origin splits, regression metrics, Seasonal Naive comparison, and
reporting conventions already established by the Ridge baseline. The first
LightGBM slice should improve the model-comparison story without pulling in
MLflow, API serving, dashboards, or model artifact persistence yet.

## Current project context

The repository already has:

- API ingestion for City of Melbourne sensor locations and hourly counts;
- immutable raw snapshots and manifests;
- validation pipelines and reports;
- PostgreSQL persistence helpers plus smoke-test coverage;
- local Prefect ingestion orchestration;
- leakage-safe feature builders in `src/urbanflow/features`;
- supervised direct multi-horizon rows with `forecast_horizon`;
- Seasonal Naive and Ridge modeling helpers;
- rolling-origin Ridge evaluation with Seasonal Naive comparison fields;
- a JSON evaluation CLI and Markdown report for Ridge;
- checked-in synthetic report examples;
- Ruff, pytest, and GitHub Actions coverage.

The missing MVP modeling piece is LightGBM. Requirements call for a global
LightGBM model trained across multiple sensors, evaluated with the same
time-based protocol as Ridge and Seasonal Naive, and eventually tracked through
MLflow.

## Selected approach

Implement LightGBM in staged slices:

1. add a local LightGBM train/predict module using the existing
   `ModelFeatureSpec`;
2. add rolling-origin LightGBM evaluation using the same windows as Ridge;
3. expose LightGBM evaluation through a CLI JSON summary;
4. render a Markdown model-comparison report;
5. only after the local model path is stable, add MLflow tracking and model
   artifacts.

This keeps the next implementation small and testable while still moving toward
the MVP requirement of comparing Seasonal Naive, Ridge, and LightGBM.

## Alternatives considered

### 1. Add LightGBM, MLflow, artifacts, and report changes together

Rejected for the next slice. That would touch dependency management,
experiment tracking, file artifacts, JSON schema, and report rendering at once.
It would be harder to test and harder to review.

### 2. Hyperparameter-tune LightGBM immediately

Rejected for the first LightGBM slice. The project needs a reproducible,
leakage-safe baseline before tuning. Hyperparameter search can be added after a
stable default LightGBM evaluation exists.

### 3. Replace Ridge evaluation with a generic evaluator first

Rejected as the first step. A generic evaluator is attractive, but a broad
refactor would risk breaking the now-stable Ridge report flow. The initial
implementation should extract only the shared pieces that LightGBM actually
needs.

## Data contract

LightGBM should consume the same supervised DataFrame contract as Ridge:

- `location_id`;
- `forecast_origin_at`;
- `target_observed_at`;
- `forecast_horizon`;
- `target`;
- `target_missing`;
- calendar, lag, rolling, weather, and missing-marker feature columns from
  `ModelFeatureSpec`.

The model must keep the existing direct multi-horizon setup: each row predicts
one target timestamp and one `forecast_horizon`. It must not recursively feed
earlier predictions into later horizons.

## Model behavior

The first LightGBM module should live at:

- `src/urbanflow/modeling/lightgbm.py`

It should provide:

- `LightGBMModelConfig`;
- `DEFAULT_LIGHTGBM_MODEL_CONFIG`;
- `FittedLightGBMModel`;
- `fit_lightgbm_model(train_frame, *, config=...)`;
- `add_lightgbm_predictions(frame, fitted_model)`.

The first configuration should be deterministic and modest:

- fixed `random_state`;
- bounded number of estimators;
- no hyperparameter search;
- no early stopping in the first slice;
- prediction column `lightgbm_prediction`;
- predictions clipped at zero to satisfy the non-negative forecast
  requirement.

The implementation should reuse `select_model_features`. It may use a small
preprocessing pipeline for categorical columns if that keeps the interface
consistent with Ridge. The first version should not add sensor-specific models;
it should be one global model trained over all locations in the training window.

## Dependency strategy

LightGBM is a core MVP model dependency, but it should be introduced only when
the implementation slice starts.

Implementation should:

- add the LightGBM package to project dependencies;
- verify local install and tests on Windows;
- keep the dependency in the main project dependencies rather than dev-only,
  because model training is runtime functionality;
- avoid optional imports that hide missing dependency errors in the training
  path.

If installation or CI compatibility fails, pause and resolve that dependency
problem directly rather than falling back to a fake model.

## Evaluation behavior

LightGBM evaluation should use the exact same `RollingOriginSplits` objects as
Ridge:

- each validation window trains only on rows before `window.train_end`;
- each evaluation window scores only rows in `[window.start, window.end)`;
- final test is evaluated once;
- Seasonal Naive predictions are computed for the same evaluation rows;
- missing Seasonal Naive history affects only Seasonal Naive metric rows, not
  LightGBM rows.

The evaluation output should include:

- LightGBM predictions;
- LightGBM overall metrics;
- LightGBM per-horizon metrics;
- Seasonal Naive overall metrics;
- Seasonal Naive per-horizon metrics;
- relative WAPE improvement versus Seasonal Naive.

Ridge comparison should be added after LightGBM can be evaluated reliably. The
first LightGBM evaluation does not need to recompute Ridge in the same command.

## CLI and report direction

The current command is Ridge-specific:

```powershell
python scripts/evaluate_ridge_baseline.py data/modeling/supervised_rows.csv --validation-months 3
```

LightGBM should get a separate first command:

```powershell
python scripts/evaluate_lightgbm_baseline.py data/modeling/supervised_rows.csv --validation-months 3
```

This avoids renaming the stable Ridge command or breaking existing examples.
Once LightGBM is stable, a later slice can add a unified comparison report that
combines Seasonal Naive, Ridge, and LightGBM.

The first LightGBM report should be explicit that scores are local evaluation
results from a supervised CSV, not production model performance.

## Testing strategy

The implementation should follow TDD in small slices.

### Model tests

Add `tests/unit/modeling/test_lightgbm.py` covering:

- missing target handling uses existing `ModelTrainingError`;
- `fit_lightgbm_model` records training row count;
- `add_lightgbm_predictions` adds `lightgbm_prediction`;
- predictions are finite numeric values;
- negative raw predictions are clipped to zero, either through a focused helper
  test or a deterministic fitted-model seam.

### Evaluation tests

Add LightGBM evaluation tests proving:

- training and evaluation rows follow the same time filters as Ridge;
- overall and per-horizon metrics are returned;
- Seasonal Naive comparison metrics are present;
- relative WAPE improvement uses
  `(seasonal_naive_wape - lightgbm_wape) / seasonal_naive_wape`;
- missing Seasonal Naive rows do not remove LightGBM prediction rows.

### CLI tests

Add CLI tests proving:

- JSON summary contains LightGBM metrics and Seasonal Naive comparison fields;
- invalid options return code `2`;
- missing input returns code `2`;
- no Seasonal Naive metric rows across all windows returns code `2`;
- conflicting Seasonal Naive panel inputs return code `2`.

### Report tests

Add report tests only after the CLI JSON shape is stable:

- report includes a LightGBM versus Seasonal Naive comparison table;
- checked-in example JSON and Markdown stay renderer-synced;
- older Ridge reports remain renderable by the existing Ridge report command.

## Out of scope for the first LightGBM baseline

- MLflow logging;
- model artifact persistence;
- feature importance plots;
- hyperparameter tuning;
- API serving;
- Streamlit/Plotly dashboard changes;
- Evidently monitoring;
- database-backed training reads;
- Docker Compose changes;
- production model-version registry.

These belong after the local LightGBM evaluation path is stable and tested.

## Acceptance criteria

The LightGBM baseline stage is complete when:

- LightGBM dependency is installed and verified locally;
- a global LightGBM model can train and predict from supervised rows;
- predictions are non-negative;
- rolling-origin evaluation uses the same windows as Ridge;
- JSON output contains LightGBM overall, horizon, Seasonal Naive, and comparison
  metrics;
- a checked-in synthetic LightGBM report example renders from its JSON summary;
- README explains how to run the LightGBM evaluation without making unsupported
  performance claims;
- full Ruff and pytest suites pass.

## Recommended implementation sequence

1. Add LightGBM dependency, model wrapper, and focused model tests.
2. Add LightGBM rolling-origin evaluation and comparison metrics.
3. Add LightGBM CLI JSON output.
4. Add Markdown report and checked-in example.
5. Add README instructions.
6. Run full repository verification and reassess whether to proceed to MLflow
   or unified multi-model comparison next.
