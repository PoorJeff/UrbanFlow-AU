# Streamlit Operations Dashboard Design

**Date:** 2026-07-23

## Goal

Add the first runnable UrbanFlow AU operations dashboard. It must present only
truthful information obtained through the existing FastAPI boundary: service
health, sensors, their history, model evaluation metadata, and a real
artifact-backed forecast when the API has been configured to serve one.

The dashboard is an operator-facing tool, not a marketing page and not a
second data-serving implementation. It must remain useful when the local API
is partially configured, and must never replace unavailable results with demo
data, stale data, or a locally computed forecast.

## Current context

UrbanFlow AU already has:

- typed FastAPI routes for `/health`, sensors, sensor history, sensor
  forecasts, and model metrics;
- optional PostgreSQL repositories behind that API boundary;
- an opt-in local LightGBM artifact provider for real direct 1–24 hour
  forecasts;
- a safe default API state with no database or artifact I/O, where health is
  `degraded` and unavailable resources are represented honestly;
- existing `httpx`, Pydantic, pandas, pytest, Ruff, and Uvicorn dependencies.

The requirements document describes a four-page final dashboard, data-drift
monitoring, and Docker Compose. Those capabilities are not all supported by
the current API or operational inputs. This slice therefore establishes the
truthful dashboard boundary and its first three operator workflows before the
separate monitoring and deployment slices.

## Decision

Build a small Streamlit application that consumes the FastAPI HTTP contract
through one typed client. The app has three pages:

1. `Overview` for health, configured data availability, active-sensor count,
   and optional evaluation metrics;
2. `Sensor Explorer` for an operator-selected, bounded history query;
3. `Forecast` for an explicitly requested 1–24 hour model forecast and its
   recent real history.

Use `URBANFLOW_DASHBOARD_API_BASE_URL` as the sole runtime connection setting.
When the variable is absent, it defaults to `http://127.0.0.1:8000` for the
documented local two-process workflow. A present-but-blank or invalid value is
a visible configuration error; it must not silently fall back to another URL.

The dashboard adds `streamlit>=1.60,<2` and `plotly>=6.9,<7` as runtime
dependencies. Streamlit 1.60.0 and Plotly 6.9.0 were the current verified
PyPI releases when this design was written; the upper bounds preserve the
project's existing major-version policy.

## Scope

### Included

1. A Streamlit entry point at `app/streamlit_app.py` and focused dashboard
   modules under `src/urbanflow/dashboard/`.
2. A synchronous, bounded-timeout `httpx` API client with one method per
   existing FastAPI endpoint.
3. Reuse of `urbanflow.api.schemas` response models for successful response
   validation; no copy of the server contract and no import of API services,
   repositories, or model providers.
4. The three pages and their truthful empty, unavailable, and successful
   states.
5. Two Plotly line charts only: selected sensor history and selected sensor
   history plus forecast.
6. Unit/UI tests with mocked HTTP, an offline Streamlit smoke, updated README
   instructions, and CI coverage appropriate to the new entry point.

### Explicit non-goals

- direct PostgreSQL, MLflow, artifact, filesystem, Melbourne API, or model
  access from the dashboard;
- a fourth `Monitoring` page, Evidently, drift reports, alerts, retraining,
  model registry integration, or automated model promotion;
- maps, daily/weekly aggregation, anomaly detection, missing-rate
  calculations, or cross-sensor aggregate flow calculations;
- selectable model comparisons or a Seasonal Naive fallback in the UI;
- login, write actions, polling, background refresh, caching that can hide
  current availability, or a dashboard-owned data cache;
- Docker Compose, container images, deployment, screenshots, or portfolio
  claims.

## Alternatives considered

### Call FastAPI over HTTP — selected

The dashboard is a real consumer of the boundary already intended for it. It
exercises the same configuration, response schemas, and error semantics that
an external operator or later deployment will use. Tests can mock this small
client without starting PostgreSQL, MLflow, or a model server.

### Reuse repositories and model code in-process

Rejected. It would bypass the serving boundary, create duplicate database and
artifact configuration paths, and make an apparently working Dashboard differ
from the API it is meant to operate beside.

### Build a static portfolio dashboard

Rejected. Static values or example forecasts would not tell an operator
whether data or a model is actually available, and would conflict with the
project's requirement to avoid misleading claims.

## Architecture

```text
operator browser
      |
      v
app/streamlit_app.py
      |
      +-- dashboard configuration
      +-- dashboard page renderers
      +-- DashboardApiClient (httpx, 5-second timeout, no retries)
                         |
                         v
                  UrbanFlow FastAPI
                    /health
                    /api/v1/sensors
                    /api/v1/sensors/{id}/history
                    /api/v1/sensors/{id}/forecast
                    /api/v1/model/metrics
```

`DashboardApiClient` owns URL construction, bounded HTTP calls, parsing of the
API's `{"error": {"code", "message", "details"}}` envelope, and successful
response validation. Page renderers receive typed results or one dashboard
error type; they do not inspect raw `httpx.Response` objects. The application
does not cache network responses or retain a prior query result in session
state, so a failed current request cannot leave old data displayed as if it
were current.

The dashboard may import Pydantic response models from
`urbanflow.api.schemas`, which is a contract-only module. It must not import
`create_app`, `ApiServices`, a repository, a database engine, or a forecast
provider.

## Runtime configuration

| Setting | Meaning | Default | Invalid state |
| --- | --- | --- | --- |
| `URBANFLOW_DASHBOARD_API_BASE_URL` | FastAPI origin used by the dashboard | `http://127.0.0.1:8000` only when absent | Present blank text, a non-HTTP(S) scheme, or an origin without a host prevents requests and shows a configuration error |

The normalized origin has trailing slashes removed. A request timeout is five
seconds and the dashboard does not retry: an operator should see a responsive
availability state rather than wait through a browser-side retry policy.

## API-client contract

The client provides these methods:

| Method | HTTP request | Successful model | Special handling |
| --- | --- | --- | --- |
| `get_health()` | `GET /health` | `HealthResult` | Both `200` and `503` with a valid health body are parsed, so `unavailable` remains an observable health state rather than a generic transport error. |
| `list_sensors(active_only=True)` | `GET /api/v1/sensors` | `SensorListResponse` | The Dashboard always uses the active-only list. |
| `get_history(location_id, start, end)` | `GET /api/v1/sensors/{id}/history` | `HistoryResponse` | Sends ISO-8601 timestamps with offsets. |
| `get_forecast(location_id, horizon)` | `GET /api/v1/sensors/{id}/forecast` | `ForecastResponse` | Accepts only `1..24`; preserves the API's prediction order. |
| `get_model_metrics()` | `GET /api/v1/model/metrics` | `ModelMetricsResponse` | `503 metrics_unavailable` is optional-page information, not a fatal Overview failure. |

For every other non-success response, the client attempts to parse the
standard error envelope and raises `DashboardApiError` with the status, code,
message, and details. A timeout/connection failure raises `api_unreachable`;
non-JSON or schema-invalid successful bodies raise `invalid_api_response`;
unrecognised HTTP failures raise `api_request_failed`. These errors are
displayed as plain operator-visible messages and never as Python tracebacks in
the page body.

## Page behavior

### Shared behavior

The sidebar identifies the configured API origin and selects one of the three
pages. Each page performs its own current request. A request is only made by a
page render or a form submission; importing the Streamlit module performs no
network I/O.

No response is fabricated. Empty lists, empty history, nullable model
versions, and unavailable components remain visible as such. All timestamps
are displayed in `Australia/Melbourne` with an explicit timezone-aware source
value retained by the client.

### Overview

`Overview` obtains current health first. When the health body is
`unavailable`, it displays that state without issuing dependent sensor or
metrics requests. Otherwise it obtains the active sensor list and presents:

- overall API health and the four components (`api_process`, `data_store`,
  `model_provider`, and `data_freshness`);
- health generation time, data cutoff, and model version when provided;
- the active sensor count from `SensorListResponse.meta.count`;
- optional evaluation metadata from `/api/v1/model/metrics`: model name,
  nullable version, MAE, RMSE, WAPE, and final test window.

`degraded` is rendered as an informative warning, not a page failure.
`unavailable`, a connection failure, or invalid health output shows a clear
error and prevents misleading dependent metrics. If the metrics endpoint
returns `metrics_unavailable`, Overview remains usable and says that
evaluation metrics are not configured.

The final requirements' global "past 24-hour total" and "forecast peak"
cards are intentionally absent. The API has no cross-sensor aggregation
contract; making a fan-out of forecasts in the browser would be expensive,
non-atomic, and would invent a second aggregation API. A later API slice must
define that source before the cards are added.

### Sensor Explorer

`Sensor Explorer` first requests active sensors. With a non-empty list, a
form lets the operator choose a sensor and an exclusive date range. The UI
turns the selected calendar dates into `Australia/Melbourne` local-midnight
timestamps. It requires `start < end` and an actual timezone-aware elapsed
interval of at most 31 days, then sends the exact ISO timestamps to the
history route. This matches the API limit even across a daylight-saving change;
it may reject a calendar-looking 31-day range if the local offset makes that
elapsed interval longer than 31 days.

The initial form values describe the most recent seven complete calendar days,
but no history request happens until submission. A successful non-empty result
is shown as a Plotly time-series line and a timestamp/count table. An empty
result says that no observations exist in the requested interval. Sensor
absence, `sensor_not_found`, `data_store_unavailable`, invalid range input,
and transport failures each display their own clear state without a chart.

The page does not make a map, infer anomalies, calculate a missing-rate, or
derive daily/weekly patterns. Those need a separately defined aggregation and
quality contract.

### Forecast

`Forecast` uses the same active sensor list and a form containing sensor and
integer horizon controls. The horizon accepts exactly `1..24` and defaults to
24. Submitting the form first obtains the one direct multi-step forecast. It
then requests the real preceding 24-hour history ending at the forecast's
`data_cutoff_at`, using `data_cutoff_at + timedelta(microseconds=1)` as the
end-exclusive timestamp so that an observation at the cutoff is included.

On complete success, the page shows:

- model name, nullable model version, generated timestamp, forecast origin,
  and data cutoff;
- a Plotly chart containing the real historical points and returned forecast
  points as separate traces;
- a prediction table in API order;
- the largest returned non-negative prediction and its target time.

If the forecast succeeds but its auxiliary history query fails, the page may
show the truthful forecast-only chart with a prominent note that recent
history is unavailable; it must not substitute stale history. If forecast
generation itself fails (`model_unavailable`, `forecast_unavailable`, storage
failure, invalid response, or transport failure), it shows no prediction
chart, peak, or fake fallback.

## Rendering and data rules

- Preserve API timestamp instants and convert only for display; do not use
  naive datetimes.
- Use exact integer counts for history and returned non-negative floats for
  forecasts. Do not round or clip an API response into a different value.
- Treat an empty response as a valid empty state only where the endpoint's
  contract allows it. Treat malformed model fields or prediction order as an
  invalid API response.
- Do not use `st.cache_data`, polling, hidden reruns, local CSV fallbacks, or
  a sample-data mode.
- Page code stays compact: data parsing and request errors belong in the API
  client; layout/wording and plotting belong in page renderers.

## Testing and verification

Routine tests must be entirely offline and require no PostgreSQL server,
artifact directory, MLflow server, running FastAPI process, or Melbourne API.

1. Configuration tests cover absent/default origin, valid normalized origin,
   and present-but-invalid settings.
2. API-client tests use `httpx.MockTransport` to cover all five successful
   contracts, health's valid `503 unavailable` body, standard error envelopes,
   timeouts/connection errors, non-JSON errors, and malformed success payloads.
3. Page/rendering tests use the mocked client and Streamlit's supported test
   harness. They cover Overview `ok`, `degraded`, `unavailable`, and metrics
   unavailable states; sensor-list/history empty and error states; history
   range boundary validation; forecast horizon boundaries, ordered output,
   auxiliary-history failure, model unavailable, and unavailable storage.
4. A bounded headless Streamlit smoke verifies the entry point starts without
   connecting to an API on module import. It must be stopped and checked for a
   ready server within a fixed timeout.
5. The full project Ruff check, Ruff format check, pytest suite, and existing
   bounded Uvicorn health smoke stay required. CI receives only deterministic,
   local checks.

## Documentation and operator workflow

README will document this local sequence:

1. Configure and start the FastAPI process, including the optional PostgreSQL,
   artifact, and metrics settings when real data is desired.
2. Optionally set `URBANFLOW_DASHBOARD_API_BASE_URL` when the API is not on
   `http://127.0.0.1:8000`.
3. Run `python -m streamlit run app/streamlit_app.py` from the project
   environment.

It will state that an unconfigured API intentionally yields empty/degraded
Dashboard states, that real forecasts require the existing API configuration,
and that the dashboard is an initial non-production operations boundary.

## Acceptance criteria

This slice is complete when:

- importing Dashboard modules performs no HTTP request, while an active
  Streamlit user session performs only the page's documented current request;
- every dashboard value is sourced from a current FastAPI response and the
  UI never shows synthetic or stale predictions;
- Overview accurately renders health and optional metrics behavior;
- Sensor Explorer validates and sends timezone-aware elapsed intervals of no
  more than 31 days, then renders only returned history;
- Forecast validates `1..24`, renders API-order direct forecasts, and
  truthfully distinguishes a missing model from missing auxiliary history;
- normal errors are readable availability/configuration messages, not a
  silent empty chart or an uncaught traceback;
- tests are fully mocked/offline, quality gates pass, and CI remains green;
- README gives an operator a reproducible API-plus-Dashboard startup path;
- no dashboard code introduces a model fallback, a direct data connection,
  monitoring, Docker, or deployment claim.
