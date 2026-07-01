# Ridge Baseline Modeling Design

Date: 2026-07-01

## Goal

Add the first trainable modeling slice for UrbanFlow AU: a leakage-safe Ridge
Regression baseline that trains and evaluates on the existing supervised
multi-horizon rows.

This slice should prove the model-training contract before heavier modeling and
MLOps work begins. It should reuse the current feature rows, rolling-origin
split utilities, metrics, and Seasonal Naive baseline. It should not add
LightGBM, MLflow, forecast serving, dashboard code, or live database extraction.

## Current project context

The repository already has:

- ingestion, validation, PostgreSQL persistence, and a local Prefect ingestion
  flow;
- `urbanflow.features` builders for hourly panels, calendar features,
  origin-anchored lags, rolling statistics, missing markers, and direct
  `forecast_horizon=1..24` supervised rows;
- `urbanflow.modeling` utilities for chronological rolling-origin splits,
  regression metrics, peak Top 10% MAE, grouped summaries, and Seasonal Naive
  predictions.

The repository does not yet have:

- a model feature matrix contract;
- a trainable model wrapper;
- scikit-learn in project dependencies;
- a model evaluation runner that fits on past rows and predicts validation or
  final-test windows;
- MLflow, LightGBM, or database-backed model training.

## Selected approach

Implement a DataFrame-first Ridge slice with three small modeling modules:

1. `urbanflow.modeling.feature_matrix` defines which supervised columns may
   enter model training, validates required columns, and documents the leakage
   exclusions.
2. `urbanflow.modeling.ridge` owns a scikit-learn Ridge pipeline for fitting
   and predicting from supervised DataFrames.
3. `urbanflow.modeling.evaluation` applies an evaluation window by fitting on
   rows before `window.train_end`, predicting rows in `[window.start,
   window.end)`, and summarizing metrics by horizon and overall.

The implementation should stay local and deterministic. Tests should use small
in-memory DataFrames, not PostgreSQL, network calls, MLflow, or large fixture
files.

## Alternatives considered

### 1. Ridge baseline first

This is the selected approach. It is the smallest trainable model slice and
keeps the project on the current leakage-safe path. It creates a reusable model
training interface before adding heavier models or experiment tracking.

### 2. Database reader first

A database reader is useful, but it would only move data around. It would not
prove that the supervised rows can train a model or beat a baseline under the
rolling-origin split contract.

### 3. Ridge, LightGBM, and MLflow together

This would advance more buzzword boxes at once, but it would mix dependency
installation, model behavior, experiment logging, and evaluation semantics. It
would be harder to debug and easier to accidentally hide a leakage or split
issue behind tooling.

## Dependency decision

Add `scikit-learn` as the first modeling dependency in this slice. Ridge,
`ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, and
`OneHotEncoder` are standard, well-tested primitives for this task.

Use a bounded dependency such as:

```toml
"scikit-learn>=1.5,<2"
```

Subsequent advanced model dependencies such as LightGBM can move into an
optional modeling extra if dependency weight becomes a packaging concern. Ridge
should be part of the main project dependency set because the repository will
import it from `urbanflow.modeling.ridge` and test it in the default suite.

## Model input contract

The Ridge slice consumes the supervised frame created by
`urbanflow.features.build_supervised_frame`.

Rows are eligible for training when:

- `target` is present;
- `target_missing` is false;
- `target_observed_at` is before the evaluation window's `train_end`.

Rows are eligible for prediction when:

- `target_observed_at` is inside the evaluation window interval;
- all required feature columns exist, even if some values are missing;
- target may be missing for raw prediction, but metric calculation must drop
  rows with missing actuals or missing predictions.

## Feature contract

The default Ridge feature set should include:

- categorical: `location_id`;
- numeric: `forecast_horizon`, `pedestrian_count`, `lag_1`, `lag_24`,
  `lag_168`, `rolling_24_mean`, `rolling_24_std`, `rolling_168_mean`,
  `rolling_168_std`, `hour`, `weekday`, `month`, `hour_sin`, `hour_cos`,
  `weekday_sin`, `weekday_cos`, `temperature`, `rainfall`, and `wind_speed`;
- boolean or marker columns cast to numeric: `pedestrian_count_missing`,
  `is_weekend`, `is_public_holiday`, `temperature_missing`,
  `rainfall_missing`, and `wind_speed_missing`.

The default Ridge feature set must exclude:

- target columns: `target`, `target_missing`;
- future timestamp columns: `target_observed_at`;
- origin timestamp columns: `forecast_origin_at`;
- baseline prediction columns: `seasonal_naive_prediction`,
  `seasonal_naive_missing`, `seasonal_naive_observed_at`;
- any evaluation output columns such as `ridge_prediction`;
- free-form columns not explicitly listed in the feature specification.

Calendar features derived from `target_observed_at` are allowed because the
future forecast timestamp is known at forecast creation time. Lag and rolling
features are allowed only because the existing feature builder anchors them at
`forecast_origin_at` instead of the target time.

## Ridge pipeline behavior

`urbanflow.modeling.ridge` should expose a small function-oriented API:

- `RidgeModelConfig`: immutable configuration for `alpha`, numeric columns,
  categorical columns, and prediction column name.
- `fit_ridge_model(train_frame, config=RidgeModelConfig())`: validates rows,
  builds the scikit-learn pipeline, and returns a fitted wrapper.
- `add_ridge_predictions(frame, fitted_model)`: returns a copy of `frame` with
  a `ridge_prediction` column.

The scikit-learn pipeline should:

- one-hot encode `location_id` with unknown categories ignored;
- impute numeric and boolean feature values with median or constant values;
- standardize numeric values before Ridge;
- fit `Ridge(alpha=config.alpha)`;
- return finite numeric predictions for eligible rows.

The wrapper should keep enough metadata for tests and downstream callers to
inspect which feature columns were used.

## Evaluation behavior

`urbanflow.modeling.evaluation` should expose:

- `evaluate_model_window(supervised_frame, window, model_config)`;
- `evaluate_rolling_origin_ridge(supervised_frame, splits, model_config)`.

For each `EvaluationWindow`:

1. Select training rows with `target_observed_at < window.train_end`.
2. Select evaluation rows with `window.start <= target_observed_at <
   window.end`.
3. Drop training rows with missing target values.
4. Fit Ridge only on the selected training rows.
5. Add `ridge_prediction` to the evaluation rows.
6. Calculate overall metrics and per-horizon metrics with the existing
   `calculate_regression_metrics` and `summarize_by_group` helpers.
7. Preserve row-level predictions so Seasonal Naive and Ridge can be compared
   using the same actual target rows.

The evaluation helpers should not persist artifacts, write files, or log to
MLflow in this slice.

## Error handling

Raise a modeling-specific `ModelTrainingError` when:

- required feature or target columns are missing;
- no training rows remain after filtering;
- no evaluation rows exist for a window;
- Ridge prediction is attempted before fitting;
- a fitted model is asked to predict from a frame missing required features.

Metric helpers already tolerate missing actual or predicted values by dropping
invalid rows. The Ridge slice should lean on that behavior rather than inventing
a parallel metric implementation.

## Testing strategy

Unit tests should cover:

- feature specification includes the intended safe columns and excludes target,
  timestamp, baseline, and output columns;
- training fails clearly when required columns are missing;
- fitting on a tiny synthetic supervised frame returns a fitted wrapper with
  stable metadata;
- prediction returns the same number of rows as the input frame and finite
  `ridge_prediction` values for valid rows;
- evaluation windows train only on rows before `train_end` and predict only
  inside the requested window;
- per-horizon metric output is produced through the existing metrics helpers;
- unknown `location_id` values at prediction time do not crash the one-hot
  encoder.

The implementation plan should continue the current repository pattern:
write failing tests first, run them to verify red, implement the minimal code,
then run focused tests and the full quality gate.

## Out of scope

This slice intentionally does not add:

- LightGBM;
- MLflow tracking;
- model artifact persistence;
- database readers for model training;
- production forecast serving;
- Streamlit or dashboard visualization;
- feature importance reports;
- hyperparameter search.

Those become safer after the Ridge contract demonstrates that supervised rows,
splits, model fitting, predictions, and metrics all work together.

## Success criteria

The implementation is successful when:

- `scikit-learn` is declared as a project dependency;
- Ridge can fit and predict from the existing supervised frame contract;
- rolling-origin evaluation produces row-level `ridge_prediction` values and
  metric summaries;
- all Ridge training and evaluation behavior is covered by focused unit tests;
- the full repository quality gate passes;
- README documentation names Ridge as the first trainable local baseline and
  keeps LightGBM, MLflow, and database-backed training as subsequent slices.
