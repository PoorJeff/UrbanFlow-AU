# Leakage-Safe Features and Evaluation Design

Date: 2026-06-30

## Goal

Add the first modeling-foundation slice for UrbanFlow AU: a deterministic,
leakage-safe feature builder, rolling-origin evaluation utilities, and a
Seasonal Naive baseline.

This slice turns the persisted hourly pedestrian data into supervised
multi-horizon rows for `forecast_horizon=1..24`. It does not train Ridge,
LightGBM, or log MLflow experiments yet. Those remain the next modeling and
MLOps slices after the data contract and leakage tests are stable.

## Current project context

The repository already has:

- ingestion for City of Melbourne sensor locations and hourly counts;
- validation reports for raw snapshots;
- PostgreSQL persistence for `sensor_dim` and `pedestrian_hourly_fact`;
- a local Prefect flow that can ingest, validate, and optionally load snapshots.

There is not yet a `urbanflow.features` or `urbanflow.modeling` package. The
feature/evaluation layer should therefore be DataFrame-first and independent of
live PostgreSQL, network calls, or MLflow. Later scripts can add thin database
readers and experiment tracking on top of these pure, testable foundations.

## Requirements covered by this slice

The full project requires time features, public-holiday features, lag features,
rolling statistics, sensor ID, weather features, missing markers, strict
time-based evaluation, Seasonal Naive, Ridge, LightGBM, and MLflow.

This slice covers the stable foundation:

- time and cyclic calendar features;
- public-holiday support through an explicit supplied holiday calendar;
- `lag_1`, `lag_24`, and `lag_168`;
- past 24-hour and 168-hour rolling mean and standard deviation;
- sensor/location ID;
- weather feature columns and missing markers without fetching weather data;
- direct multi-horizon target rows for horizons 1 through 24;
- rolling-origin validation and final-test window definitions;
- MAE, RMSE, WAPE, per-sensor WAPE, calendar-slice errors, peak Top 10% MAE;
- Seasonal Naive predictions and metric evaluation.

Ridge, LightGBM, MLflow, weather ingestion, and production forecast serving are
intentionally deferred.

## Selected approach

Create two small packages:

- `urbanflow.features` builds the hourly modeling frame.
- `urbanflow.modeling` owns split definitions, metrics, and baselines.

The public API stays function-oriented and pandas-based so the behavior is easy
to test with tiny in-memory frames before connecting it to large database
extracts.

## Alternatives considered

### 1. Feature and evaluation foundation first

This is the selected approach. It protects the project from the most expensive
modeling failure mode: accidental time leakage. It also gives every future model
the same supervised rows, split windows, and metrics.

### 2. Train Ridge and LightGBM immediately

This would look faster, but it would mix feature correctness, split correctness,
model dependencies, and experiment tracking in one change. If a metric looked
good, it would be hard to know whether the model improved or the data leaked.

### 3. Build a SQL feature mart first

This would be closer to production scale, but it would slow down iteration
before the exact feature contract is proven. A SQL feature mart can be added
later once the pandas implementation and tests define the expected behavior.

## Data contract

The base feature builder accepts a pandas DataFrame with at least:

- `location_id`: integer sensor/location identifier;
- `observed_at`: timezone-aware hourly timestamp in Melbourne local time;
- `pedestrian_count`: non-negative observed count.

Optional columns may include:

- `temperature`;
- `rainfall`;
- `wind_speed`.

Weather values are only safe for model training when they represent information
available at the forecast origin, such as a forecast snapshot. Until a weather
source with availability timestamps exists, this slice will keep weather columns
nullable and expose `*_missing` markers. It will not silently use future actual
weather as if it were known at prediction time.

Public holidays are supplied as an explicit collection of dates. The feature
builder will not download or hard-code holiday calendars in this slice.
Hourly timestamps must be on exact hour boundaries. Daylight-saving transitions
should be handled with timezone-aware pandas ranges rather than converting to
timezone-naive local time.

## Forecast-row convention

Each supervised row represents one sensor, one forecast origin, and one horizon:

- `forecast_origin_at`: the latest timestamp whose pedestrian count may be used
  by the model;
- `forecast_horizon`: integer from 1 through 24;
- `target_observed_at`: `forecast_origin_at + forecast_horizon hours`;
- `target`: pedestrian count at `target_observed_at`.

Known future calendar features describe `target_observed_at`, because hour of
day, weekday, weekend status, month, and public-holiday status are knowable
before the forecast is issued.

Lag and rolling features are anchored to the forecast origin, not to each target
horizon. This means every horizon row for the same sensor and origin shares the
same historical target features. The model may receive `forecast_horizon`, but
it may not receive true counts from any hour after `forecast_origin_at`.

## Feature definitions

For each `(location_id, forecast_origin_at)`:

- `lag_1`: pedestrian count at `forecast_origin_at`;
- `lag_24`: pedestrian count at `forecast_origin_at - 23 hours`;
- `lag_168`: pedestrian count at `forecast_origin_at - 167 hours`;
- `rolling_24_mean`: mean over the 24 known hours ending at
  `forecast_origin_at`;
- `rolling_24_std`: standard deviation over the same 24 known hours;
- `rolling_168_mean`: mean over the 168 known hours ending at
  `forecast_origin_at`;
- `rolling_168_std`: standard deviation over the same 168 known hours;
- `location_id`: retained as the sensor identifier;
- `forecast_horizon`: retained as a direct multi-step feature;
- `hour`, `weekday`, `month`, `is_weekend`, `is_public_holiday`:
  calendar features for `target_observed_at`;
- `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`:
  cyclic encodings for the target hour and weekday;
- `temperature`, `rainfall`, `wind_speed`:
  optional exogenous columns, nullable in this slice;
- `temperature_missing`, `rainfall_missing`, `wind_speed_missing`:
  boolean missing markers;
- `target_missing`:
  boolean marker for missing target values after panel completion.

The lag naming is intentionally anchored to the first forecasted hour. For a
one-hour-ahead prediction, `lag_1` is the previous known hour. For longer
horizons, `lag_1` remains the latest value known at the forecast origin, not the
future value one hour before that later target.

## Hourly panel handling

The feature builder first creates a complete hourly panel per `location_id` over
the available timestamp range. Missing observed counts remain missing and are
marked; they are not imputed in this slice.

Rows that lack required lag or rolling history remain in intermediate frames
with missing feature values. Training/evaluation helpers may drop rows with
missing required features, missing targets, or missing baseline predictions, but
they must report the number of dropped rows so coverage problems are visible.

Duplicate `(location_id, observed_at)` observations are invalid for the feature
builder. The database layer already upserts by this key, so duplicates in a
DataFrame input indicate a caller-side data-quality problem and should raise a
clear error.

## Split design

The splitter never uses random splits.

Given a completed hourly panel:

1. Infer the final complete calendar month from the data coverage. A complete
   month means the panel's time range covers every hour from the first day
   00:00 through the last day 23:00 for that calendar month. Missing target
   values inside the month are tracked by missing markers and do not by
   themselves make the month incomplete.
2. Reserve that final complete month as the final test month.
3. Use the previous three complete months as rolling-origin validation when
   available. If only one or two complete months are available before the test
   month, use the available months and report the reduced count.
4. For validation month `M`, the training window ends before `M` starts and the
   validation window covers origins whose target timestamps fall inside `M`.
5. The final test month is evaluated once after model selection.

Seasonal Naive does not fit parameters, but it will still be evaluated through
the same split objects so future Ridge and LightGBM models can reuse the exact
windows.

## Metrics

Metric utilities will accept actual and predicted values after missing
predictions have been filtered.

Required metrics:

- MAE;
- RMSE;
- WAPE, with `None` returned when the actual-value denominator is zero;
- per-sensor WAPE;
- grouped MAE/RMSE/WAPE by hour;
- grouped MAE/RMSE/WAPE by weekday/weekend;
- peak Top 10% MAE, where peak rows are selected by actual count within the
  evaluated frame.

Metrics must not fabricate model performance. Reports should include row counts,
prediction coverage, and any undefined WAPE groups.

## Seasonal Naive baseline

Seasonal Naive predicts the pedestrian count from the same sensor at the same
hour one week earlier:

`prediction = count(target_observed_at - 168 hours)`

The baseline is non-negative by construction when source counts are
non-negative. If the one-week-prior value is missing, the baseline prediction is
missing and the row is excluded from metric numerators and denominators while
coverage is reported.

## Error handling

Feature functions should fail fast on:

- missing required input columns;
- duplicate `(location_id, observed_at)` rows;
- non-hourly or timezone-naive `observed_at` values;
- unsupported forecast horizons;
- invalid split requests, such as no complete month available.

They should not fail solely because a target, lag, rolling statistic, or weather
value is missing. Those missing values are part of the modeling data contract
and are handled by downstream row filters and coverage reporting.

## Testing strategy

The automated tests for this slice should be small and deterministic.

Feature tests:

- calendar and cyclic encoding tests for known timestamps;
- public-holiday flag tests with an injected holiday date;
- dense panel creation and missing marker tests;
- duplicate key and timezone validation tests;
- lag and rolling tests using a monotonic sequence where expected values are
  obvious;
- a leakage guard that mutates future counts after a forecast origin and proves
  all lag/rolling features for that origin stay unchanged;
- direct multi-horizon tests proving `forecast_horizon=1..24` rows share the
  same origin-anchored lag/rolling features while targets move forward.

Split tests:

- no random split path exists;
- final test month is the last complete month;
- rolling validation windows never overlap with their training windows;
- final test rows are not included in validation windows.

Metric and baseline tests:

- MAE/RMSE/WAPE numeric examples;
- WAPE zero-denominator behavior;
- per-sensor and calendar-slice grouping;
- peak Top 10% MAE selection;
- Seasonal Naive one-week-prior prediction;
- Seasonal Naive missing-history coverage reporting.

The final verification gate remains:

```powershell
$env:PYTHONPATH='src'
python -m ruff check . --no-cache
python -m ruff format --check .
python -m pytest
```

## Proposed file boundaries

### `src/urbanflow/features/__init__.py`

Exports public feature-building functions and result dataclasses.

### `src/urbanflow/features/calendar.py`

Owns calendar, public-holiday, weekend, and cyclic encoding helpers.

### `src/urbanflow/features/hourly_panel.py`

Builds and validates complete per-sensor hourly panels from raw hourly
observations.

### `src/urbanflow/features/lagged.py`

Computes origin-anchored lag and rolling features per sensor.

### `src/urbanflow/features/supervised.py`

Combines panel, calendar, lagged, optional weather, and target columns into
direct multi-horizon supervised rows.

### `src/urbanflow/modeling/splits.py`

Defines split dataclasses and rolling-origin/final-test month selection.

### `src/urbanflow/modeling/metrics.py`

Computes aggregate and grouped regression metrics with explicit undefined-WAPE
behavior.

### `src/urbanflow/modeling/baselines.py`

Implements Seasonal Naive prediction and evaluation helpers.

### `tests/unit/features/`

Contains focused unit tests for calendar features, panel completion, lag/rolling
features, supervised horizon rows, and leakage guards.

### `tests/unit/modeling/`

Contains focused unit tests for split windows, metrics, and Seasonal Naive.

## Out of scope for this slice

This slice will not add:

- Ridge Regression;
- LightGBM;
- MLflow tracking;
- feature importance plots;
- model artifact files;
- production forecast API routes;
- Streamlit or dashboard UI;
- weather data ingestion or weather forecast acquisition;
- public-holiday data downloading;
- SQL feature marts or materialized views.

## Acceptance criteria

The slice is complete when:

- `urbanflow.features` can turn hourly observations into direct
  `forecast_horizon=1..24` supervised rows;
- lag and rolling tests prove no feature uses data after `forecast_origin_at`;
- calendar, public-holiday, weather-missing-marker, and sensor ID features are
  represented in the supervised rows;
- rolling-origin and final-test split utilities enforce chronological windows;
- Seasonal Naive predictions and metrics are available through
  `urbanflow.modeling`;
- automated tests cover leakage, split boundaries, metrics, and baseline
  behavior;
- Ruff and pytest pass on `main`;
- no model-performance claims are made yet.
