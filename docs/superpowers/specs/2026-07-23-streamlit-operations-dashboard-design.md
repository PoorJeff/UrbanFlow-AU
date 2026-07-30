# Streamlit Guided Operations Dashboard Design

**Date:** 2026-07-23

**Updated:** 2026-07-26

## Goal

Add the first runnable UrbanFlow AU guided operations dashboard. It must orient
an operator around one selected sensor and its returned observations before
offering deeper history and forecast investigations. All information continues
to come only from the existing FastAPI boundary: service configuration signals,
sensors, their history, model evaluation metadata, and a real artifact-backed
forecast when the API has been configured to serve one.

The dashboard is an operator-facing tool with a calm, human-centred entry
experience, not a marketing page, explanatory AI, or a second data-serving
implementation. It must remain useful when the local API is partially
configured, and must never replace unavailable results with demo data, stale
data, a locally computed forecast, or an invented explanation of pedestrian
behaviour.

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
truthful dashboard boundary and its first three linked workflows before the
separate monitoring and deployment slices.

## Decision

Build a small Streamlit application that consumes the FastAPI HTTP contract
through one typed client. The app has three pages arranged as one journey:

1. `Today` is the home page. It asks the operator to select one returned
   sensor, then explicitly load that location's bounded observations and, when
   available, its direct 24-hour forecast. It leads with a factual timeline,
   then offers clear next questions rather than a wall of technical cards.
2. `Explore` is the selected sensor's bounded history query. It pre-fills the
   focus sensor but still lets the operator choose a different sensor and date
   range.
3. `Forecast` is the selected sensor's explicitly requested 1–24 hour model
   forecast and its recent real history. It pre-fills the focus sensor but
   never requests a forecast merely because the page was opened.

`Today` owns the visual hierarchy; `Explore` and `Forecast` are its deliberate
extensions. The only application-owned cross-page data context is
`selected_location_id`, so that the location travels between pages. The app
must not retain a health, sensor-list, history, forecast, metrics response, or
plot data in session state.

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
4. The three linked pages and their truthful first-visit, empty, unavailable,
   and successful states.
5. Two Plotly line-chart forms only: selected sensor history, and selected
   sensor history plus a separately labelled forecast trace.
6. Deterministic factual captions derived only from response fields, visible
   non-colour status wording, accessible control labels, and a quiet
   data-and-model transparency area.
7. Unit/UI tests with mocked HTTP, an offline Streamlit smoke, updated README
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
- generative explanations, causal claims, crowding recommendations, inferred
  neighbourhoods, route context, confidence intervals, or a sample-data mode;
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
      +-- focus context (selected_location_id only)
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
error type; they do not inspect raw `httpx.Response` objects. The only
application-owned cross-page data context in `st.session_state` is
`selected_location_id`; no API response or plot data is retained there. The
application does not cache network responses or retain a prior query result in
session state, so a failed current request or a newly selected location cannot
leave old data displayed as if it were current.

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
| `get_model_metrics()` | `GET /api/v1/model/metrics` | `ModelMetricsResponse` | `503 metrics_unavailable` is optional transparency information, not a fatal Today-page failure. |

For every other non-success response, the client attempts to parse the
standard error envelope and raises `DashboardApiError` with the status, code,
message, and details. A timeout/connection failure raises `api_unreachable`;
non-JSON or schema-invalid successful bodies raise `invalid_api_response`;
unrecognised HTTP failures raise `api_request_failed`. These errors are
displayed as plain operator-visible messages and never as Python tracebacks in
the page body.

## Page behavior

### Shared behavior

A light navigation names the three pages `Today`, `Explore`, and `Forecast`.
It identifies the configured API origin and, after a location is chosen,
visibly names the current focus location. The persisted context is only the
integer `selected_location_id`; a sensor list and every detailed response are
read again when needed. If a newly returned active list no longer contains the
persisted ID, the app clears the focus rather than repeatedly requesting an
unknown sensor. Importing a Streamlit module performs no network I/O.

Each page starts by obtaining current health and the active sensor list when
health is not `unavailable`. Health is a configuration and availability signal,
not proof that a database, artifact, model, or data-freshness component is
ready: the existing default API legitimately reports `degraded` and
`unconfigured` components. `unavailable`, a connection failure, or invalid
health output blocks dependent requests and shows a readable state.

No response is fabricated. Empty lists, empty history, nullable model
versions, and unavailable components remain visible as such. All timestamps
are displayed in `Australia/Melbourne` with an explicit timezone-aware source
value retained by the client. A selection alone never loads observations or a
forecast; a clearly labelled form action performs the detailed request.

### Today

`Today` is the first page and begins with an invitation to choose an active
sensor returned by the API. Until a sensor is chosen and the operator submits
`Load this location`, it shows no observations, forecast line, numeric example,
or city-wide statement. If the active list is empty, it says that no active
locations are currently returned and does not invent a Melbourne catalog.

When the operator explicitly loads a selected location, the page performs a
bounded snapshot for that one ID:

1. It requests the direct 24-hour forecast for the same ID.
2. If forecast succeeds, it sets
   `end = data_cutoff_at + timedelta(microseconds=1)` and
   `start = end - timedelta(hours=24)`, then requests that exact offset-aware
   history interval. This is the only history trace that may share a chart with
   the returned forecast trace.
3. If forecast fails with `model_unavailable` or `forecast_unavailable`, it
   sets `end = now(Australia/Melbourne)` and
   `start = end - timedelta(hours=24)`, then requests that exact offset-aware
   history interval so that a returned-observations-only state remains possible.

The page shows a selected-location heading, exact sensor name/description,
and a factual timeline. A non-empty history result may state the latest
*returned* observation and its timestamp. A successful forecast may state the
model name/version, generation and cutoff timestamps, and the largest
*returned* prediction with its target timestamp. It labels observed and
forecast traces separately; forecast uses a different line style as well as a
label.

If the forecast fails with `model_unavailable` or `forecast_unavailable`, the
page may still show the independently returned history trace and a clear
`Forecast unavailable` panel containing the API message. It shows no forecast
trace, forecast maximum, or fallback prediction. If history fails, its panel
states that observations could not be returned and it shows no history trace.
If forecast succeeds but the matching history is a valid empty list, it shows
the forecast-only trace and says that no observations were returned for the
matching interval; it does not imply a continuous history.
For `data_store_unavailable`, `sensor_not_found`, invalid response, or
transport failure, the page presents the applicable availability state and
never continues with stale output.

Below the primary result, two explicit question-style calls to action carry
the selected ID to the other pages: `How has this location changed?` opens
`Explore`; `What returned forecast is available next?` opens `Forecast`.
The lower-prominence `Data and model context` area may display health
components, health generation time, optional cutoff/model version, active
sensor count, and a user-triggered evaluation-summary view. A
`metrics_unavailable` result is an honest optional state. Evaluation MAE,
RMSE, and WAPE are labelled as historical evaluation context, never as the
current selected location's or serving forecast's accuracy.

The final requirements' global total, ranking, and city-wide forecast cards
are intentionally absent. The API has no cross-sensor aggregation contract;
making a fan-out of forecasts in the browser would be expensive, non-atomic,
and would invent a second aggregation API. A later API slice must define that
source before such cards are added.

### Explore

`Explore` requests active sensors and pre-fills its selector from
`selected_location_id` when the ID is still present in the returned list. A
form lets the operator choose a sensor and an exclusive date range. The UI
turns the selected calendar dates into `Australia/Melbourne` local-midnight
timestamps. It requires `start < end` and an actual timezone-aware elapsed
interval of at most 31 days, then sends the exact ISO timestamps to the
history route. This matches the API limit even across a daylight-saving change;
it may reject a calendar-looking 31-day range if the local offset makes that
elapsed interval longer than 31 days.

Changing a location updates `selected_location_id` but clears the current
render's result; no plot or caption for the earlier location remains visible.
The initial form values describe the most recent seven complete calendar days,
but no history request happens until submission. A successful non-empty result
is shown as a Plotly time-series line and a timestamp/count table. An empty
result says that no observations were returned in the selected interval.
Sensor absence, `sensor_not_found`, `data_store_unavailable`, invalid range
input, and transport failures each display their own clear state without a
chart.

The page does not make a map, infer anomalies, calculate a missing-rate, or
derive daily/weekly patterns. Those need a separately defined aggregation and
quality contract.

### Forecast

`Forecast` requests active sensors and pre-fills its selector from
`selected_location_id` when valid. Its form contains sensor and integer horizon
controls. The horizon accepts exactly `1..24` and defaults to 24. Changing a
location updates the shared focus ID and clears the current render's output.
Opening the page or changing a selector does not request a forecast; only
submission obtains the one direct multi-step forecast. It then requests the
real preceding 24-hour history ending at the forecast's `data_cutoff_at`, using
`data_cutoff_at + timedelta(microseconds=1)` as the end-exclusive timestamp so
that an observation at the cutoff is included.

On complete success, the page shows:

- model name, nullable model version, generated timestamp, forecast origin,
  and data cutoff;
- a Plotly chart containing returned historical points and returned forecast
  points as separately labelled, visually distinct traces;
- a prediction table in API order;
- the largest returned non-negative prediction and its target time.

If the forecast succeeds but its auxiliary history query fails, the page may
show the truthful forecast-only chart with a prominent note that recent history
is unavailable; it must not substitute stale history. If forecast generation
itself fails (`model_unavailable`, `forecast_unavailable`, storage failure,
invalid response, or transport failure), it shows no prediction chart,
prediction maximum, or fake fallback.

## Rendering and data rules

- Preserve API timestamp instants and convert only for display; do not use
  naive datetimes.
- Use exact integer counts for history and returned non-negative floats for
  forecasts. Do not round or clip an API response into a different value.
- Treat an empty response as a valid empty state only where the endpoint's
  contract allows it. Treat malformed model fields or prediction order as an
  invalid API response.
- Human-centred copy is deterministic and factual. It may identify the chosen
  sensor, a returned interval, the latest returned observation, a returned
  forecast's highest value, or API timestamps and status. It must not call a
  source live, fresh, accurate, busy, rising, healthy, city-wide, or causal
  unless that exact claim is represented by the response contract.
- Every status has visible text; colour is never its sole carrier. Chart traces
  have a visible legend and labels, and observed and forecast values use
  distinct line styles in addition to colour. Controls have explicit labels
  and charts keep a readable timestamp/count table alternative.
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
   harness. They cover Today first-visit guidance, empty catalog, `degraded`,
   `unavailable`, explicit selected-location loading, returned history plus
   forecast, forecast plus a valid empty history, history-only model-unavailable,
   metrics unavailable, and no stale location content after a selector change.
   They also cover sensor-list and
   history empty/error states; history range boundary validation; focus-context
   prefill on Explore and Forecast; forecast horizon boundaries, ordered
   output, auxiliary-history failure, model unavailable, and unavailable
   storage.
4. Rendering assertions cover factual captions, visible text for non-success
   states, labelled controls, and non-colour-only observed-versus-forecast
   distinction.
5. A bounded headless Streamlit smoke verifies the entry point starts without
   connecting to an API on module import. It must be stopped and checked for a
   ready server within a fixed timeout.
6. The full project Ruff check, Ruff format check, pytest suite, and existing
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
Dashboard states, that users first choose one returned location before loading
its observations, that real forecasts require the existing API configuration,
and that the dashboard is an initial non-production operations boundary.

## Acceptance criteria

This slice is complete when:

- importing Dashboard modules performs no HTTP request, while an active
  Streamlit user session performs only the page's documented current request;
- every dashboard value is sourced from a current FastAPI response and the
  UI never shows synthetic or stale predictions;
- Today guides a first-time user to choose a returned location, makes detailed
  requests only after explicit submission, and renders truthful history,
  forecast, and unavailable combinations without a fabricated number or
  explanation;
- `selected_location_id` travels from Today to Explore and Forecast, while
  response data never travels through session state and a changed location
  cannot leave an earlier location's result visible;
- data and model context accurately renders health as a configuration signal
  and treats optional metrics as evaluation context rather than serving
  accuracy;
- Explore validates and sends timezone-aware elapsed intervals of no
  more than 31 days, then renders only returned history;
- Forecast validates `1..24`, renders API-order direct forecasts, and
  truthfully distinguishes a missing model from missing auxiliary history;
- normal errors are readable availability/configuration messages, not a
  silent empty chart, colour-only signal, or an uncaught traceback;
- tests are fully mocked/offline, quality gates pass, and CI remains green;
- README gives an operator a reproducible API-plus-Dashboard startup path;
- no dashboard code introduces a model fallback, a direct data connection,
  monitoring, Docker, or deployment claim.
