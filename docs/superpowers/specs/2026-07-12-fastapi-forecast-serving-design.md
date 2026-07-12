# FastAPI Forecast Serving Design

Date: 2026-07-12

## Goal

Add the first FastAPI serving layer for UrbanFlow AU's pedestrian-demand
forecasting workflow.

The serving layer should expose health, sensor, history, forecast, and model
metric endpoints with typed Pydantic schemas and clear error behavior. The
first implementation should be a local, testable API boundary. It should not
claim production forecast quality until model artifacts and database-backed
serving reads are intentionally added.

## Current project context

The repository already has:

- City of Melbourne ingestion and validation helpers;
- PostgreSQL persistence models, loaders, migrations, and smoke coverage;
- Prefect-based local ingestion orchestration;
- leakage-safe supervised feature builders;
- Seasonal Naive, Ridge, and LightGBM local baselines;
- rolling-origin evaluation summaries and Markdown reports;
- MLflow tracking for existing evaluation artifacts;
- no committed model artifacts;
- no model registry or explicit model-loading contract;
- no FastAPI runtime dependency yet.

The requirements call for:

- `GET /health`;
- `GET /sensors`;
- `GET /history/{location_id}`;
- `GET /forecast/{location_id}?horizon=24`;
- `GET /model/metrics`.

For this project, keep `GET /health` unversioned for runtime probes and expose
business endpoints under `/api/v1`. This keeps the demo API explicit and leaves
room for future response-shape changes.

## Selected approach

Implement serving in staged slices:

1. add a FastAPI app factory, typed schemas, dependency seams, and health route;
2. add sensor and history routes behind repository interfaces;
3. add forecast route behind a `ForecastService` interface that requires an
   explicit model provider;
4. add model-metrics route backed by local evaluation/MLflow summary metadata;
5. wire database-backed repositories and model artifact loading only after the
   route contracts are stable.

Place API code under the installable package:

```text
src/urbanflow/api/
├── __init__.py
├── app.py
├── dependencies.py
├── errors.py
├── schemas.py
├── services.py
└── routers/
    ├── health.py
    ├── sensors.py
    ├── forecasts.py
    └── models.py
```

This keeps imports consistent with the rest of `src/urbanflow`. Docker or local
commands can later point Uvicorn at `urbanflow.api.app:create_app` or an
application object exported from that module.

## API contract

### `GET /health`

Purpose: show service readiness without requiring a forecast request.

Response fields:

- `status`: `ok`, `degraded`, or `unavailable`;
- `service`: service name;
- `version`: package/API version;
- `generated_at`: UTC timestamp;
- `components`: component health records for API process, model provider,
  data store, and data freshness;
- `model_version`: current model version when loaded;
- `data_cutoff_at`: newest observation timestamp when known.

Behavior:

- return `200` when the API process can respond, even if individual components
  are degraded;
- component-level failures appear in `components`;
- only return `503` if the API cannot initialize enough to answer health.

### `GET /api/v1/sensors`

Purpose: list forecastable sensor locations.

Query parameters:

- `active_only`: boolean, default `true`.

Response:

- `data`: list of sensors with `location_id`, `sensor_name`,
  `sensor_description`, `status`, `latitude`, and `longitude`;
- `meta`: count and `active_only`.

Behavior:

- return an empty list if no sensors are configured;
- do not contact the Melbourne API at request time;
- use repository reads from PostgreSQL or a configured local fixture.

### `GET /api/v1/sensors/{location_id}/history`

Purpose: return historical pedestrian counts for one sensor and time range.

Query parameters:

- `start`: inclusive ISO-8601 timestamp;
- `end`: exclusive ISO-8601 timestamp.

Response:

- `location_id`;
- `start`;
- `end`;
- `data`: ordered hourly records with `observed_at` and `pedestrian_count`.

Validation:

- `location_id` must exist;
- `start` must be before `end`;
- the first implementation may cap the range, for example to 31 days, to keep
  demo responses bounded.

### `GET /api/v1/sensors/{location_id}/forecast`

Purpose: return a 1-to-24 hour forecast for one sensor.

Query parameters:

- `horizon`: integer from `1` to `24`, default `24`;
- optional future parameter `model_version` once model registry support exists.

Response fields:

- `location_id`;
- `model_name`;
- `model_version`;
- `generated_at`;
- `forecast_origin_at`;
- `data_cutoff_at`;
- `horizon_hours`;
- `predictions`: ordered rows with `forecast_horizon`, `target_at`, and
  `predicted_count`;
- optional future fields `prediction_interval_lower` and
  `prediction_interval_upper`.

Behavior:

- use direct multi-horizon forecasting; do not recursively feed earlier
  predictions into later horizons;
- predictions must be non-negative;
- if no explicit model provider is loaded, return `503 model_unavailable`
  rather than generating fake predictions;
- if a fallback Seasonal Naive provider is configured later, the response must
  state `model_name=seasonal_naive` and include a degraded component status.

### `GET /api/v1/model/metrics`

Purpose: expose current model evaluation metadata for the dashboard and API
consumers.

Response fields:

- `model_name`;
- `model_version`;
- `evaluation_source`;
- `final_test_window`;
- `metrics`: at minimum MAE, RMSE, WAPE, Seasonal Naive WAPE, and relative WAPE
  improvement when available;
- `mlflow_run_id` and `mlflow_tracking_uri` when known;
- `report_path` when backed by a local evaluation report.

Behavior:

- metrics come from checked evaluation summaries or MLflow metadata, not from
  ad-hoc API request logs;
- if metrics are unavailable, return `503 metrics_unavailable` unless the
  endpoint is explicitly configured with local demo metadata.

## Error behavior

Use project-level error codes with semantic HTTP statuses:

| Code | HTTP status | Trigger |
| --- | --- | --- |
| `validation_error` | `422` | invalid horizon, timestamp, or query shape |
| `sensor_not_found` | `404` | unknown `location_id` |
| `history_range_invalid` | `422` | `start >= end` or range exceeds configured cap |
| `model_unavailable` | `503` | forecast requested before a model provider is loaded |
| `metrics_unavailable` | `503` | model metrics are not configured |
| `data_store_unavailable` | `503` | repository cannot read sensor/history data |

Preferred error shape:

```json
{
  "error": {
    "code": "model_unavailable",
    "message": "No forecast model is configured for serving.",
    "details": []
  }
}
```

FastAPI/Pydantic validation may produce framework validation details, but
project-raised errors should use the shape above.

## Dependency and configuration strategy

Add dependencies only when the implementation slice starts:

- runtime: `fastapi>=0.139,<1`;
- runtime command support: `uvicorn[standard]>=0.51,<1`;
- tests can use FastAPI's `TestClient` or `httpx` ASGI transport.

These bounds are based on a 2026-07-12 PyPI check: FastAPI `0.139.0` and
Uvicorn `0.51.0` were the current releases. Re-check package compatibility
when the implementation slice starts.

Configuration should stay explicit:

- `URBANFLOW_MODEL_PATH`: optional path to a local model bundle;
- `URBANFLOW_API_METRICS_PATH`: optional path to an evaluation summary JSON;
- `URBANFLOW_DATABASE_URL`: optional database URL for repository-backed reads;
- default test configuration should use in-memory fakes, not a live database.

Do not make the API train models on startup. Startup should load already
created artifacts or fail the model component cleanly.

## Service boundaries

The first implementation should keep route handlers thin:

- schemas validate request and response shapes;
- routers translate HTTP inputs into service calls;
- services own forecast, history, and metrics behavior;
- repositories own database or fixture reads;
- model providers own model loading and prediction.

Suggested interfaces:

- `SensorRepository.list_sensors(active_only: bool)`;
- `HistoryRepository.get_history(location_id, start, end)`;
- `ForecastService.forecast(location_id, horizon)`;
- `ModelMetadataProvider.get_metrics()`;
- `HealthService.check()`.

This makes failure paths cheap to unit-test and keeps future PostgreSQL,
MLflow, and model-registry integrations outside the route layer.

## Testing strategy

Use focused tests before broad integration:

1. app factory test proving routes mount and OpenAPI generation works;
2. health route tests for `ok`, `degraded`, and model-unavailable states;
3. sensor route tests for non-empty, empty, and repository failure cases;
4. history route tests for valid range, invalid range, unknown sensor, and
   bounded output ordering;
5. forecast route tests for valid horizon, horizon validation, unknown sensor,
   missing model provider, and non-negative ordered predictions;
6. model metrics route tests for available and unavailable metadata.

Default tests must not require PostgreSQL, network access, MLflow server,
trained model artifacts, or local `mlruns/`.

## Alternatives considered

### 1. Serve forecasts directly from training CLIs

Rejected. The evaluation CLIs are batch commands that print JSON summaries.
Serving should load an explicit artifact or provider and answer request-time
queries without retraining or rerunning backtests.

### 2. Add database, model artifact loading, and all endpoints in one slice

Rejected for the first API slice. That would mix runtime dependencies,
repository wiring, artifact serialization, route contracts, and dashboard needs
at once. The first slice should establish the HTTP and schema boundary.

### 3. Put API code in a top-level `api/` package immediately

Rejected for now. The existing project is an installable `src/urbanflow`
package. Keeping API code under `src/urbanflow/api` avoids duplicate import
roots and simplifies pytest imports.

### 4. Return fake forecasts when the model artifact is missing

Rejected. A portfolio API should be honest about model availability. Tests can
use fake providers, but runtime responses should return `503 model_unavailable`
unless an explicit fallback provider is configured and disclosed.

## Acceptance criteria

The first FastAPI serving stage is complete when:

- FastAPI and Uvicorn dependencies are added intentionally;
- an app factory exposes `/health` and versioned `/api/v1` routes;
- request and response schemas are typed with Pydantic;
- forecast responses include model version, generation time, data cutoff time,
  horizon, and non-negative predictions;
- invalid sensors, invalid ranges, missing model provider, and unavailable
  metrics return clear errors;
- OpenAPI docs are generated by FastAPI;
- README documents a local `uvicorn` command and example API calls;
- default tests pass without PostgreSQL, network access, MLflow server, or
  committed model artifacts;
- full Ruff and pytest suites pass.

## Recommended implementation sequence

1. Add FastAPI dependencies, app factory, error schema, and `/health`.
2. Add sensor and history schemas, repository protocols, fake repositories, and
   route tests.
3. Add forecast service protocol, fake model provider tests, and
   `/api/v1/sensors/{location_id}/forecast`.
4. Add model metrics provider and `/api/v1/model/metrics`.
5. Document local API usage in README.
6. Reassess whether the next slice should be model artifact persistence,
   PostgreSQL-backed API reads, or Streamlit integration.
