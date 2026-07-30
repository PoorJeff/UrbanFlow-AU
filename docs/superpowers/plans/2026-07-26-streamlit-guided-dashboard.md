# Streamlit Guided Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` for every behavior change, and make a focused commit after each completed task.

**Goal:** Deliver the first runnable, truthful Streamlit operations dashboard
for UrbanFlow AU.  It guides an operator from one selected sensor's current
returned information to bounded history and direct forecast investigation,
using only the existing FastAPI HTTP contract.

**Architecture:** Add a small `urbanflow.dashboard` package with a validated
runtime origin, one synchronous typed `httpx` client, pure focus/time/chart and
request-orchestration helpers, and thin Streamlit page renderers.  The entry
point creates the client only for the current render and dispatches the
`Today`, `Explore`, and `Forecast` views.  It neither owns data access nor
retains API responses: FastAPI remains the single serving boundary.

**Tech Stack:** Python 3.11+ (current worktree: Python 3.12), Streamlit 1.x,
Plotly 6.x, `httpx`, existing FastAPI/Pydantic response models, `pytest`,
`streamlit.testing.v1.AppTest`, Ruff, and a bounded headless Streamlit smoke.

## Global Constraints

- Work only in the existing isolated worktree on
  `codex/streamlit-operations-dashboard`; do not modify the root worktree
  while this slice is being built.
- Add exactly `streamlit>=1.60,<2` and `plotly>=6.9,<7` as runtime
  dependencies.  Do not add a dashboard framework, database driver, model
  package, or front-end build system.
- First repair the worktree's incomplete editable install.  The root `.venv`
  is healthy and must not be deleted, rebuilt, or pointed at the worktree.
  Do not delete/rebuild the worktree `.venv` unless a bounded editable reinstall
  reproducibly fails and its captured logs establish that this is necessary.
- `URBANFLOW_DASHBOARD_API_BASE_URL` is the dashboard's only connection setting.
  Its absent value means exactly `http://127.0.0.1:8000`; a present blank value,
  non-HTTP(S) scheme, or URL without a host is a visible configuration error.
  Normalize a valid origin by removing its trailing slash.
- The dashboard may make only bounded HTTP calls through `DashboardApiClient`:
  five-second timeout, no retries, no background refresh, no polling, and no
  `st.cache_data`.  Importing a dashboard module must perform no HTTP I/O.
- Reuse only the success schemas in `urbanflow.api.schemas`.  Dashboard code
  must not import `create_app`, `ApiServices`, repositories, PostgreSQL,
  MLflow, artifact-loading, model-provider, filesystem, or Melbourne Open Data
  code.
- The only application-owned cross-page data context is the integer
  `selected_location_id`.  Framework-owned widget state is allowed, but no
  health, sensor-list, history, forecast, metrics response, table rows, or
  Plotly figure may be stored in `st.session_state`.
- All detailed results start from an explicit form submission.  A selection,
  navigation change, or page load alone must not request history or forecast.
  If an active sensor list no longer contains the stored focus ID, clear it.
- Preserve all offset-aware timestamp instants.  Convert only for display with
  `Australia/Melbourne`; never use a naive `datetime` as a query boundary.
- Never synthesize a sensor, observation, prediction, model version, stale
  result, fallback forecast, aggregate, explanation, or a claim such as
  “live”, “fresh”, “accurate”, “busy”, “rising”, “healthy”, “city-wide”, or
  causal.  Every value and fact caption must come from the current API result.
- Every non-success state needs visible text, not colour alone.  Observed and
  forecast traces require a legend, distinct labels, and different line styles;
  tables remain a readable alternative to charts.
- Explicitly exclude maps, city-wide totals/rankings, monitoring, Evidently,
  retraining, model comparison/fallback, writes, logins, Docker/Compose,
  deployment, sample data, and dashboard-owned data caching.
- Routine tests must be fully offline.  They may use `httpx.MockTransport`,
  fakes, and `AppTest`, but not a running FastAPI process, PostgreSQL, MLflow,
  an artifact directory, Melbourne Open Data, or external network access.
- Keep all changes surgical.  Do not alter existing FastAPI semantics or model
  behavior to suit the dashboard.

---

### Task 1: Declare the dashboard runtime and repair the isolated environment

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/unit/dashboard/__init__.py`

**Implementation:**

1. Add the two dependency bounds to `[project].dependencies`, keeping the
   alphabetical/project-local ordering already used in `pyproject.toml`.
2. Create the dashboard test package so later focused invocations have a stable
   location.  Do not create application modules yet.
3. From this worktree, run one bounded editable install using its own Python.
   Capture stdout/stderr to temporary files; if it exceeds five minutes, stop
   only that child process, print the captured logs, and diagnose before trying
   a different repair.  Do not use `PYTHONPATH` as a workaround.
4. Verify that `urbanflow`, Streamlit, and Plotly import from this worktree,
   and that `pip show urbanflow-au` reports an installed editable distribution.
5. Run the existing baseline quality gate before adding dashboard behavior.

**Bounded install command:**

```powershell
$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "The isolated worktree Python is missing: $python"
}

$stdout = Join-Path $env:TEMP "urbanflow-dashboard-pip.stdout.log"
$stderr = Join-Path $env:TEMP "urbanflow-dashboard-pip.stderr.log"
$process = Start-Process `
    -FilePath $python `
    -ArgumentList @("-m", "pip", "install", "-e", ".[dev]") `
    -WorkingDirectory (Get-Location) `
    -WindowStyle Hidden `
    -PassThru `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr

if (-not $process.WaitForExit(300000)) {
    Stop-Process -Id $process.Id -Force
    Get-Content -LiteralPath $stdout, $stderr -ErrorAction SilentlyContinue
    throw "Editable install exceeded the five-minute bound."
}
if ($process.ExitCode -ne 0) {
    Get-Content -LiteralPath $stdout, $stderr -ErrorAction SilentlyContinue
    throw "Editable install failed with exit code $($process.ExitCode)."
}

& $python -c "import urbanflow, streamlit, plotly; from pathlib import Path; print(Path(urbanflow.__file__).resolve())"
& $python -m pip show urbanflow-au
& $python -m ruff check .
& $python -m ruff format --check .
& $python -m pytest
```

**Verify:**

- The printed `urbanflow` path is under
  `D:\Github项目\UrbanFlow-AU\.worktrees\streamlit-operations-dashboard\src`.
- All three imports succeed without `PYTHONPATH`.
- The existing quality gate passes before the new feature work begins.

**Commit:**

```text
build: add dashboard runtime dependencies
```

---

### Task 2: Add validated dashboard configuration and a typed FastAPI client

**Files:**
- Create: `src/urbanflow/dashboard/__init__.py`
- Create: `src/urbanflow/dashboard/config.py`
- Create: `src/urbanflow/dashboard/errors.py`
- Create: `src/urbanflow/dashboard/client.py`
- Create: `tests/unit/dashboard/conftest.py`
- Create: `tests/unit/dashboard/test_config.py`
- Create: `tests/unit/dashboard/test_client.py`

**Interfaces:**

```python
# src/urbanflow/dashboard/config.py
DASHBOARD_API_BASE_URL_ENV_VAR = "URBANFLOW_DASHBOARD_API_BASE_URL"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


class DashboardConfigError(ValueError):
    """Raised when the explicitly supplied dashboard origin is unusable."""


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    api_base_url: str


def load_dashboard_config(
    environ: Mapping[str, str] | None = None,
) -> DashboardConfig: ...


# src/urbanflow/dashboard/errors.py
@dataclass(frozen=True, slots=True)
class DashboardApiError(RuntimeError):
    status_code: int | None
    code: str
    message: str
    details: tuple[object, ...]


# src/urbanflow/dashboard/client.py
class DashboardApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 5.0,
    ) -> None: ...

    def close(self) -> None: ...

    def get_health(self) -> HealthResult: ...
    def list_sensors(self, *, active_only: bool = True) -> SensorListResponse: ...
    def get_history(
        self, location_id: int, *, start: datetime, end: datetime
    ) -> HistoryResponse: ...
    def get_forecast(
        self, location_id: int, *, horizon: int
    ) -> ForecastResponse: ...
    def get_model_metrics(self) -> ModelMetricsResponse: ...
```

**TDD cycle:**

1. Write `test_config.py` first.  Cover an absent setting (and only an absent
   setting) choosing the default, removal of trailing slashes, and explicit
   rejection of blank text, `ftp://...`, `http:///path`, and a URL without a
   host.  Use an injected mapping so tests do not mutate process environment.
2. Write `test_client.py` before production code.  Build one `httpx.Client`
   around `httpx.MockTransport`; inject it into `DashboardApiClient`.  Verify
   each documented request shape and `model_validate` parsing for the five
   existing successful schemas:

   | Client method | Request to assert |
   | --- | --- |
   | `get_health()` | `GET /health` |
   | `list_sensors()` | `GET /api/v1/sensors?active_only=true` |
   | `get_history()` | `GET /api/v1/sensors/{id}/history` with offset-aware ISO `start` and `end` |
   | `get_forecast()` | `GET /api/v1/sensors/{id}/forecast?horizon=...` |
   | `get_model_metrics()` | `GET /api/v1/model/metrics` |

3. Include the failure contract in those tests:

   - A valid `503` `/health` body is parsed as `HealthResult`, so a genuine
     `status="unavailable"` reaches the UI.
   - Every other error response tries the existing
     `{"error": {"code", "message", "details"}}` envelope and becomes
     `DashboardApiError` with the original status and details.
   - A timeout or connection failure becomes `api_unreachable` with no status.
   - A non-JSON or Pydantic-invalid 2xx body becomes `invalid_api_response`.
   - An unrecognised non-success response becomes `api_request_failed`.
   - `get_history` rejects a naive timestamp or `start >= end` before issuing
     HTTP.  `get_forecast` rejects a horizon outside `1..24` before issuing
     HTTP.
   - Construction/import performs no HTTP request, does not retry, and
     `close()` does not close an injected client owned by a test caller.

4. Add malformed-but-type-shaped 2xx fixtures and assert they become
   `invalid_api_response`, rather than being displayed as if authoritative:

   - every page-facing response timestamp is offset-aware (health generated
     time/cutoff, history interval/points, forecast metadata/predictions, and
     metrics final-test window);
   - a history or forecast response `location_id` matches the requested ID;
   - forecast `horizon_hours` equals the requested horizon; its predictions
     remain in the API's required order with horizons exactly `1..N` once each;
     there are exactly `N` predictions; each prediction is finite and
     non-negative; and no response is sorted, repaired, or clipped by the
     dashboard;
   - a supplied metrics interval has `start < end` and its displayed numeric
     values are finite.

5. Implement only the config parser, one generic private request/parse helper,
   a compact private semantic-validator layer, and the five thin public
   methods.  Successful bodies must first be validated with the source models
   from `urbanflow.api.schemas`, then pass the semantic checks above; do not
   duplicate server Pydantic classes.  Use `datetime.isoformat()` only after
   the history-boundary guard has confirmed offset awareness and ordering.  The
   client owns and closes only a client it creates itself.
6. Add compact fixtures in `tests/unit/dashboard/conftest.py` for valid
   response payloads and the fixed offset-aware timestamps needed by later
   dashboard tests.  Fixtures must contain no fake runtime service.

**Focused verification:**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/dashboard/test_config.py tests/unit/dashboard/test_client.py
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/dashboard tests/unit/dashboard
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/dashboard tests/unit/dashboard
& .\.venv\Scripts\python.exe -m pytest
```

**Commit:**

```text
feat(dashboard): add typed FastAPI dashboard client
```

---

### Task 3: Build pure focus, time, chart, and request-orchestration helpers

**Files:**
- Create: `src/urbanflow/dashboard/context.py`
- Create: `src/urbanflow/dashboard/time_utils.py`
- Create: `src/urbanflow/dashboard/charts.py`
- Create: `src/urbanflow/dashboard/snapshots.py`
- Create: `tests/unit/dashboard/test_context.py`
- Create: `tests/unit/dashboard/test_time_utils.py`
- Create: `tests/unit/dashboard/test_charts.py`
- Create: `tests/unit/dashboard/test_snapshots.py`

**Interfaces:**

```python
# context.py
SELECTED_LOCATION_ID_KEY = "selected_location_id"

def get_selected_location_id(
    session_state: MutableMapping[str, object],
) -> int | None: ...

def set_selected_location_id(
    session_state: MutableMapping[str, object], location_id: int
) -> None: ...

def clear_selected_location_if_missing(
    session_state: MutableMapping[str, object],
    sensors: Sequence[SensorResponse],
) -> int | None: ...


# time_utils.py
MELBOURNE_TIME_ZONE = ZoneInfo("Australia/Melbourne")

def melbourne_now() -> datetime: ...
def format_melbourne_timestamp(value: datetime) -> str: ...
def local_midnight(value: date) -> datetime: ...
def validate_history_interval(start: datetime, end: datetime) -> str | None: ...


# snapshots.py
@dataclass(frozen=True, slots=True)
class TodaySnapshot:
    forecast: ForecastResponse | None
    forecast_error: DashboardApiError | None
    history: HistoryResponse | None
    history_error: DashboardApiError | None


def load_today_snapshot(
    client: DashboardApiClient,
    *,
    location_id: int,
    now: Callable[[], datetime] = melbourne_now,
) -> TodaySnapshot: ...


@dataclass(frozen=True, slots=True)
class ForecastSnapshot:
    forecast: ForecastResponse | None
    forecast_error: DashboardApiError | None
    history: HistoryResponse | None
    history_error: DashboardApiError | None


def load_forecast_snapshot(
    client: DashboardApiClient,
    *,
    location_id: int,
    horizon: int,
) -> ForecastSnapshot: ...


# charts.py
def build_history_figure(history: HistoryResponse) -> go.Figure: ...
def build_forecast_figure(
    *, history: HistoryResponse | None, forecast: ForecastResponse
) -> go.Figure: ...
```

**TDD cycle:**

1. Add pure mapping tests showing that only a valid integer focus is returned,
   `set_selected_location_id` records only the selected ID, and a focus missing
   from a newly returned active catalog is removed.  Do not add response data
   or extra dashboard application state to the mapping.
2. Add time tests for Melbourne conversion while preserving an input instant,
   local-midnight offset-awareness, `start < end`, and an *elapsed* maximum of
   31 days across a daylight-saving transition.  A naive interval must return
   a validation message rather than be silently repaired.
3. Add chart tests before chart code.  Assert an observed-only history figure
   has a readable `Observed` trace, a combined/forecast-only figure labels
   `Observed` and `Forecast`, and forecast uses a different dash style rather
   than relying only on colour.  Assert x values preserve response order and
   the returned values; do not aggregate, round, or re-sort predictions.
4. Add snapshot tests with a recording fake client.  They must assert the exact
   request sequence and boundaries:

   - Today calls `get_forecast(location_id, horizon=24)` first.
   - On forecast success, it calls history with
     `end = data_cutoff_at + timedelta(microseconds=1)` and
     `start = end - timedelta(hours=24)`.
   - On `model_unavailable` or `forecast_unavailable`, it calls history once
     with `end = now()` in Melbourne and the preceding 24 elapsed hours, and
     retains no forecast result.
   - On any other forecast error, it makes no dependent history request.
   - A successful forecast plus a valid empty history remains a valid
     forecast-only snapshot; a history error is represented separately.
   - The Forecast workflow makes the same cutoff-aligned 24-hour history call
     only after a successful forecast, retains forecast when its auxiliary
     history call fails, and makes no history call when forecast fails.

5. Implement these as small pure modules.  No helper may import Streamlit,
   issue HTTP directly, place a response in state, or interpret health as a
   proof of data/model availability.

**Focused verification:**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/dashboard/test_context.py tests/unit/dashboard/test_time_utils.py tests/unit/dashboard/test_charts.py tests/unit/dashboard/test_snapshots.py
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/dashboard tests/unit/dashboard
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/dashboard tests/unit/dashboard
& .\.venv\Scripts\python.exe -m pytest
```

**Commit:**

```text
feat(dashboard): add truthful dashboard primitives
```

---

### Task 4: Compose the Streamlit shell and the guided Today page

**Files:**
- Create: `src/urbanflow/dashboard/application.py`
- Create: `src/urbanflow/dashboard/pages/__init__.py`
- Create: `src/urbanflow/dashboard/pages/shared.py`
- Create: `src/urbanflow/dashboard/pages/today.py`
- Create: `app/streamlit_app.py`
- Create: `tests/unit/dashboard/test_today.py`
- Create: `tests/unit/dashboard/test_streamlit_app.py`

**Interfaces:**

```python
# application.py
def create_dashboard_client(config: DashboardConfig) -> DashboardApiClient: ...
def main() -> None: ...

# pages/shared.py
@dataclass(frozen=True, slots=True)
class PageContext:
    health: HealthResult | None
    sensors: SensorListResponse | None
    error: DashboardApiError | None

def load_page_context(client: DashboardApiClient) -> PageContext: ...
def render_api_error(error: DashboardApiError, *, heading: str) -> None: ...

# pages/today.py
def render_today(client: DashboardApiClient) -> None: ...
```

**Implementation behavior:**

1. Make `app/streamlit_app.py` only import and call `main()` under
   `if __name__ == "__main__"`.  Import must have no HTTP request.
2. In `main()`, load configuration and render a readable configuration error
   if it is invalid.  On a valid origin, create one client for this render,
   ensure it is closed in `finally`, and render the initial `Today` workflow.
   Do not import, reference, or expose `Explore`/`Forecast` renderers until
   their tasks create them; Task 5 adds the second navigation choice and Task 6
   completes the final three-page navigation.  Navigation widget state is
   framework state only; do not retain page results.
3. `load_page_context` calls health first.  If it returns `status="unavailable"`
   or raises a dashboard error, it must not request sensors or detailed data.
   Otherwise request only the active sensor list.  A `degraded` health result
   is rendered as a configuration/availability signal, not as evidence that
   a model or data store is ready.  Clear a persisted focus ID if absent from
   the newly returned active list.
4. `Today` renders an inviting initial state, active-sensor selector, and a
   labelled `Load this location` form button.  The first visit and a selector
   change show no history, forecast, maximum, or city-wide statement.
5. Submission updates `selected_location_id`, invokes `load_today_snapshot`,
   and renders only its current result:

   - on full data, show the selected sensor's returned name/description, a
     factual latest returned observation, a combined labelled Plotly timeline,
     the history table, model name/version, generated/cutoff timestamps, and
     the largest *returned* prediction with target time;
   - on forecast success plus valid empty history, show a forecast-only chart
     with an explicit “no observations were returned for the matching interval”
     message;
   - on `model_unavailable` or `forecast_unavailable`, show a visible
     `Forecast unavailable` message and, if independently returned, a
     history-only chart/table; never show a forecast trace, maximum, or
     fallback;
   - on history failure, show a visible observations-unavailable message and
     never retain/render an earlier history result;
   - on other forecast/data/sensor/transport/validation errors, show the
     applicable readable state without a stale result.

6. Keep `Data and model context` visually quieter and include health components,
   health generation time, nullable model version/cutoff, and active sensor
   count only when returned.  Task 5 and Task 6 respectively add the two
   cross-page question actions only after their destinations exist.
7. Add a user-triggered `View historical model evaluation` action inside that
   context.  It alone calls `get_model_metrics()`.  Render MAE/RMSE/WAPE as
   “historical evaluation context”, never current sensor or serving accuracy;
   show `metrics_unavailable` as an optional visible state.

**TDD cycle:**

1. Use `streamlit.testing.v1.AppTest` with a fake recording client in a
   `from_function` page harness; do not start FastAPI.  Use
   `AppTest.from_file` on the absolute `app/streamlit_app.py` path with a
   deliberately invalid `URBANFLOW_DASHBOARD_API_BASE_URL` to execute the real
   entrypoint offline, assert no `at.exception`, and assert the visible
   configuration error.  This is the entrypoint-execution evidence; a server
   readiness probe alone is not enough.
2. First prove module/entrypoint import sends no request.  Then test first
   visit, valid `degraded`, valid `unavailable`, a transport/invalid health
   error, and an empty sensor catalog.
3. Assert selecting a sensor alone sends no history/forecast request; the
   submit action is the only trigger.  Assert the recording client sees
   forecast then history on normal Today load.
4. Add UI assertions for all truthful Today result combinations from Task 3,
   metrics unavailable, zero metrics calls before its explicit action,
   observable labels, factual wording, and clearing old sensor result after a
   new selector choice.  After a successful render, assert that no
   application-owned `history`, `forecast`, `metrics`, `figure`, `table`, or
   response-payload key exists in session state: `selected_location_id` is the
   only such data key (framework widget keys are allowed).  Assert tests use
   text and calls, not brittle pixel positions or private Streamlit HTML.
5. Implement the minimum shell, shared context, and page rendering to make
   those tests pass.  Use the pure helpers from Task 3 rather than putting
   request parsing or time arithmetic inside Streamlit callbacks.

**Focused verification:**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/dashboard/test_today.py tests/unit/dashboard/test_streamlit_app.py
& .\.venv\Scripts\python.exe -m ruff check app src/urbanflow/dashboard tests/unit/dashboard
& .\.venv\Scripts\python.exe -m ruff format --check app src/urbanflow/dashboard tests/unit/dashboard
& .\.venv\Scripts\python.exe -m pytest
```

**Commit:**

```text
feat(dashboard): add guided Today workflow
```

---

### Task 5: Add the selected-sensor Explore workflow

**Files:**
- Create: `src/urbanflow/dashboard/pages/explore.py`
- Create: `tests/unit/dashboard/test_explore.py`
- Modify: `src/urbanflow/dashboard/application.py`
- Modify: `src/urbanflow/dashboard/pages/today.py`

**Interface:**

```python
def render_explore(client: DashboardApiClient) -> None: ...
```

**Implementation behavior:**

1. Use the shared current page context, then pre-fill the selector from
   `selected_location_id` only if that current active list contains it.
2. Render a labelled form with sensor, start date, end date, and an explicit
   history-load submit control.  Default dates represent the most recent seven
   complete Melbourne calendar days, but opening the page must not query
   history.
3. Turn submitted dates into offset-aware Melbourne local midnights using
   `local_midnight`.  Use `validate_history_interval` before calling the API:
   require `start < end` and actual elapsed duration no greater than 31 days.
   Preserve the exact timestamps sent to the client, including DST behavior.
4. A selector change updates only `selected_location_id`; it clears the
   current render's detailed output, so a prior sensor's chart/caption cannot
   remain visible.  A submit invokes exactly one `get_history` call.
5. On a non-empty successful response, render selected sensor identity, the
   factual returned interval, a labelled observed-history figure, and a
   timestamp/count table.  On an empty success, say only that no observations
   were returned in that selected interval.  On `sensor_not_found`,
   `data_store_unavailable`, invalid response, transport error, or invalid
   local interval, render distinct readable text and no chart.
6. Extend the navigation to `Today` and `Explore`, and add the first
   question-style Today action, `How has this location changed?`.  It navigates
   with only `selected_location_id`; Explore must fetch its own current health,
   catalog, and any later submitted history.  It must not receive a chart,
   table, health result, or response object from Today.

**TDD cycle:**

1. Add `AppTest` cases for focus prefill/clear, no initial request, and no
   request when changing a selector or dates before submit.
2. Add boundary cases for start equal/end, reverse range, a 31-day elapsed
   range, and a local 31-calendar-day range crossing DST whose elapsed value
   exceeds the API limit.  Assert a blocked range never calls the client.
3. Add submit-path tests for exact offset-aware request values, non-empty and
   empty history, unknown sensor, store unavailable, and transport/invalid
   response.  Assert the old output disappears when sensor selection changes.
4. Test the Today-to-Explore navigation carries the selected ID but no detailed
   response state.  After Explore succeeds, assert the same no-response-cache
   session-state invariant used by Today.
5. Implement the small page renderer and two-page navigation; leave API
   parsing, chart construction, and shared context in their existing layers.

**Focused verification:**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/dashboard/test_explore.py
& .\.venv\Scripts\python.exe -m ruff check app src/urbanflow/dashboard tests/unit/dashboard
& .\.venv\Scripts\python.exe -m ruff format --check app src/urbanflow/dashboard tests/unit/dashboard
& .\.venv\Scripts\python.exe -m pytest
```

**Commit:**

```text
feat(dashboard): add selected-sensor history exploration
```

---

### Task 6: Add the direct multi-horizon Forecast workflow

**Files:**
- Create: `src/urbanflow/dashboard/pages/forecast.py`
- Create: `tests/unit/dashboard/test_forecast.py`
- Modify: `src/urbanflow/dashboard/application.py`
- Modify: `src/urbanflow/dashboard/pages/today.py`

**Interface:**

```python
def render_forecast(client: DashboardApiClient) -> None: ...
```

**Implementation behavior:**

1. Use the shared current page context and current focus semantics from
   Explore.  Render a labelled sensor selector and integer horizon input with
   default `24` and strict inclusive bounds `1..24`.
2. A page load or control change must not request a forecast.  A form submit
   records the selected ID and invokes `load_forecast_snapshot` exactly once.
3. On a successful forecast, request only its matching preceding real history:
   `end = data_cutoff_at + timedelta(microseconds=1)` and
   `start = end - timedelta(hours=24)`.  Preserve the response's prediction
   order in the chart and table; do not sort it by target time or recalculate
   anything.
4. On complete success, render the returned model name, nullable version,
   generated time, origin, and cutoff; a labelled observed/forecast chart;
   a recent returned-history timestamp/count table when history exists; a
   prediction table in API order; and the largest returned non-negative
   prediction with its returned target time.  This gives both chart traces a
   readable table alternative.
5. If auxiliary history fails after a forecast succeeds, show the truthful
   forecast-only chart/table and an explicit history-unavailable note.  If
   forecast itself fails for a missing model, forecast unavailable, store
   failure, sensor absence, invalid body, or transport error, show no
   prediction chart/table/maximum/fallback and no stale earlier result.
6. Complete the final `Today` / `Explore` / `Forecast` navigation and add the
   second Today question action, `What returned forecast is available next?`.
   It carries only the selected ID.  Forecast must request its own current page
   context and submitted result; no Today response may cross the page boundary.

**TDD cycle:**

1. Add `AppTest` tests for focus prefill/clear, default horizon, boundaries
   `1` and `24`, and client-side prevention of `0` and `25` before HTTP.
2. Assert initial load/control changes do not call forecast and a form submit
   does.  Assert forecast success yields exact cutoff-aligned history bounds.
3. Add cases for full success, ordered prediction values/table, nullable model
   version, valid forecast plus auxiliary-history failure, `model_unavailable`,
   `forecast_unavailable`, store failure, sensor not found, and connection or
   schema errors.  Check the absence of prediction visuals and maxima on every
   forecast-generation failure.  Verify the chart's observed history has its
   own table alternative, and the forecast-only branch has only the prediction
   table plus an explicit history-unavailable state.
4. Test Today-to-Forecast navigation carries the selected ID but no detailed
   result, and assert the no-response-cache session-state invariant after a
   successful Forecast render.
5. Implement only the page renderer plus final shell dispatch.  Reuse the
   client, focus state, snapshot and chart helpers; do not add a model fallback
   or a second forecast path.

**Focused verification:**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/dashboard/test_forecast.py
& .\.venv\Scripts\python.exe -m ruff check app src/urbanflow/dashboard tests/unit/dashboard
& .\.venv\Scripts\python.exe -m ruff format --check app src/urbanflow/dashboard tests/unit/dashboard
& .\.venv\Scripts\python.exe -m pytest
```

**Commit:**

```text
feat(dashboard): add truthful forecast investigation
```

---

### Task 7: Add offline entrypoint smoke coverage and operator documentation

**Files:**
- Create: `tests/unit/dashboard/test_streamlit_smoke.py`
- Modify: `tests/unit/dashboard/test_charts.py`
- Modify: `README.md`

**Implementation behavior:**

1. Extend rendering tests where needed to make the accessibility/data rules
   executable: visible non-success messages, explicit control labels, factual
   caption phrases, legend labels, and non-colour observed/forecast distinction.
   Do not test subjective visual layout by brittle markup snapshots.
2. Add a single bounded subprocess *server-startup* smoke test.  Entrypoint
   execution is already proved offline by Task 4's `AppTest.from_file`; this
   process test proves only that Streamlit can bind and report ready.  It must
   select an available loopback port, start:

   ```text
   <worktree-python> -m streamlit run app/streamlit_app.py
       --server.headless=true
       --server.address=127.0.0.1
       --server.port=<free-port>
       --browser.gatherUsageStats=false
   ```

   Poll only `http://127.0.0.1:<free-port>/_stcore/health` for at most 15
   seconds.  Do not open the application root in this smoke, because that can
   initiate a Streamlit session.  Redirect both stdout and `stderr` into one
   temporary logfile rather than unread pipes; detect a child early exit on
   each polling pass and include the closed logfile text in the assertion
   failure.  In `finally`,
   use `terminate → wait(5 seconds) → kill → wait`, then remove the temporary
   logfile.  A port-selection race must fail within the same fixed deadline,
   never hang or silently fall back to a fixed port.
3. Update README with a truthful local two-process workflow:

   ```powershell
   # terminal 1: start/configure the existing FastAPI service
   .\.venv\Scripts\python.exe -m uvicorn urbanflow.api.app:app --reload

   # terminal 2: optional only when the API is on another origin
   $env:URBANFLOW_DASHBOARD_API_BASE_URL = "http://127.0.0.1:8000"
   .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
   ```

   Explain that the default unconfigured API honestly produces degraded/empty
   states, a user must choose a returned active location before detailed data
   loads, a real forecast requires existing API data/model configuration, and
   the dashboard is an initial non-production operations boundary.  Describe
   `Today → Explore / Forecast`; do not add deployment or portfolio claims.
4. Update the project roadmap wording only where it still says Streamlit is
   wholly future work.  Keep Evidently/monitoring/deployment explicitly future
   work and do not rewrite unrelated README sections.

**TDD cycle:**

1. Write and run the server-startup smoke with the entry point present; confirm
   it fails cleanly if the command cannot start, then make it pass with the
   final options/logging/cleanup.  Keep the Task 4 `AppTest.from_file` assertion
   as the proof that the script itself executes without a traceback.
2. Run all dashboard tests using mocked HTTP and verify no test requires a
   remote service.
3. Review README commands against actual paths and the default origin/config
   behavior before committing.

**Focused verification:**

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/dashboard
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
git diff --check
```

**Commit:**

```text
docs: document Streamlit dashboard workflow
```

---

### Task 8: Perform the branch review, verification, and mainline handoff

**Files:**
- Review only: all files changed in `main...HEAD`

**Review checklist:**

1. Compare the branch with
   `docs/superpowers/specs/2026-07-23-streamlit-operations-dashboard-design.md`
   and this plan.  Confirm every selected requirement is covered and that no
   explicit non-goal appears in the diff.
2. Inspect imports to confirm dashboard code reaches data only through
   `DashboardApiClient` and imports only API success schemas.  Search for
   forbidden shortcuts and caches:

   ```powershell
   rg -n "(st\.cache|requests\.|psycopg|sqlalchemy|mlflow|lightgbm|MelbourneApi|Seasonal|sample data|fallback)" app src/urbanflow/dashboard
   ```

   Investigate a match rather than suppressing it; expected false positives in
   user-facing documentation must still not imply a forbidden runtime path.
3. Confirm the only dashboard-written business state key is
   `selected_location_id`, detailed results are not held across reruns, and
   each failure path shows visible text with no fabricated chart or number.
4. Run the fresh worktree verification gate, including the existing local
   Uvicorn health smoke and the new Streamlit smoke in pytest:

   ```powershell
   & .\.venv\Scripts\python.exe -m ruff check .
   & .\.venv\Scripts\python.exe -m ruff format --check .
   & .\.venv\Scripts\python.exe -m pytest
   git status --short --branch
   ```

5. Review `git diff main...HEAD` for accidental unrelated changes.  If clean
   and all gates pass, merge the local `codex/streamlit-operations-dashboard`
   branch into `main` according to `docs/development_workflow.md`, rerun the
   same quality gate on `main`, then push only `main` if `origin` is configured.
   Never push the `codex/*` branch.  If a network action fails, retain the
   verified local commit(s), capture the bounded command/error, and report the
   exact local state instead of repeatedly hanging on a push.

**Handoff evidence:**

- List the final commit(s), worktree/main status, quality-gate results, and
  whether the mainline push actually succeeded.
- State the real visible outcome: a local three-page Streamlit dashboard that
  consumes the existing FastAPI contract and truthfully exposes unavailable
  states; it is not a deployed product, monitor, or fallback forecasting
  service.
