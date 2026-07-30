# Runtime Readiness and Serving E2E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make configured FastAPI health truthful and add a bounded, opt-in serving smoke that proves PostgreSQL, a local artifact, Uvicorn, and the Dashboard HTTP client work together.

**Architecture:** Keep the public FastAPI routes and response schemas unchanged. Add one read-only latest-observation seam to the PostgreSQL adapter, then use a RuntimeHealthService created by the app factory to turn explicit configuration and request-time repository results into component states. Keep the E2E process lifecycle in a dedicated smoke module; it uses an isolated schema and a temporary artifact, but routine tests mock all external processes and services.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic 2, SQLAlchemy 2, psycopg 3, pandas, LightGBM, joblib, httpx, Uvicorn, Streamlit typed client, pytest, Ruff, GitHub Actions.

## Global Constraints

- Preserve GET /health, all five existing public API paths, the HealthResult schema, and the injected ApiServices(health=...) seam.
- No database connection may occur while create_default_services or create_app constructs the default app. The only new readiness query occurs inside a health request.
- Add only URBANFLOW_API_MAX_DATA_AGE_HOURS. It is optional when blank or absent and otherwise must be a positive base-10 integer; invalid values fail app construction.
- Never describe a cutoff as fresh merely because it exists. Freshness is available only when a configured age threshold accepts an aware, non-future cutoff.
- Return the database latest observation as data_cutoff_at only when it is offset-aware. A naive cutoff must not cause a health 500 or be returned as a valid timestamp.
- A configured database probe failure makes health unavailable and returns HTTP 503. Missing data, a missing threshold, stale data, a missing artifact, or an invalid artifact remain degraded HTTP 200 states when health itself can respond.
- Model availability is tri-state: no usable artifact is unconfigured, a loaded artifact is available with its manifest model_version, and an explicitly configured artifact that fails validation/loading is unavailable.
- No health response, smoke JSON result, error message, or log tail may expose a database URL, secret, artifact path, or raw exception details.
- The smoke accepts only an explicit --database-url or URBANFLOW_SMOKE_DATABASE_URL; it must never fall back to URBANFLOW_DATABASE_URL.
- The smoke creates and drops only a validated schema it created. It must use a schema-scoped child database URL and never write to public.
- The child Uvicorn process uses sys.executable, loopback, one worker, no reload, a temporary log file, monotonic deadlines, and terminate then kill cleanup. Never leave a server process running after a failure.
- The smoke does not start Streamlit. It exercises DashboardApiClient over real HTTP; existing Streamlit tests remain offline rendering coverage.
- Do not add Docker, Compose, a migration, an API route, a Dashboard page, weather, monitoring, metrics changes, training-data construction, model registry, prediction persistence, or user-facing synthetic forecasts.
- Keep routine pytest fully offline. The real PostgreSQL/Uvicorn smoke is manual and opt-in, not a CI job.
- Follow docs/development_workflow.md: work only on codex/runtime-readiness-serving-e2e, use TDD, make conventional commits, merge and push only main after verification, and never push the codex branch.

## Repository Map

| Path | Responsibility after this slice |
| --- | --- |
| src/urbanflow/api/services.py | DataReadinessRepository, data-age parsing, RuntimeHealthService, and health status calculation. |
| src/urbanflow/api/postgres.py | Existing read adapter plus one latest-observation query. |
| src/urbanflow/api/app.py | Environment-aware runtime-health construction and artifact tri-state wiring. |
| src/urbanflow/api/serving_e2e_smoke.py | Isolated PostgreSQL schema, temporary artifact, bounded Uvicorn lifecycle, HTTP assertions, and cleanup. |
| scripts/smoke_test_serving_e2e.py | Thin executable wrapper around the smoke main function. |
| tests/unit/api/test_health.py | Health service and HTTP status behavior. |
| tests/unit/api/test_app.py | Default app factory laziness, environment parsing, and artifact state wiring. |
| tests/unit/api/test_postgres_repositories.py | Latest-observation query and error mapping. |
| tests/unit/api/test_serving_e2e_smoke.py | Offline smoke parser, child configuration, lifecycle, and cleanup tests. |
| src/urbanflow/dashboard/pages/today.py | Accurate readiness wording only. |
| tests/unit/dashboard/test_today.py | Visible wording assertion only. |
| README.md | Health configuration matrix and opt-in E2E operator instructions. |

## Preflight

Run these commands from the isolated worktree before Task 1. The interpreter
must import urbanflow from this worktree, not the main checkout. If a fresh
worktree environment cannot finish its dependency installation within the
bounded command timeout, stop its child processes and repair the environment
before code changes; do not accidentally test main source.

~~~powershell
$ErrorActionPreference = "Stop"
& .\.venv\Scripts\python.exe -c "import pathlib, urbanflow; print(pathlib.Path(urbanflow.__file__).resolve())"
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest -q
git status --short --branch
~~~

Expected: the import path is under this worktree, Ruff passes, pytest passes,
and the branch contains only the already committed specification and plan
documents.

---

### Task 1: Add a read-only latest-observation repository seam

**Files:**

- Modify: src/urbanflow/api/postgres.py
- Modify: tests/unit/api/test_postgres_repositories.py

**Interfaces:**

~~~python
class PostgresSensorHistoryRepository:
    def get_latest_observed_at(self) -> datetime | None: ...
~~~

The method selects the maximum PedestrianHourlyFact.observed_at, returns None
for an empty table, closes its short-lived session in every path, and maps all
SQLAlchemyError values to DataStoreUnavailableError.

- [ ] **Step 1: Write the focused failing repository tests**

Extend FakeSession with a scalar() method that records its statement and can
return None, a datetime, or raise a configured SQLAlchemyError. Add these
tests:

~~~python
def test_get_latest_observed_at_returns_aware_maximum_and_uses_max_query() -> None:
    cutoff = datetime(2026, 7, 30, 9, tzinfo=UTC)
    session = FakeSession([cutoff])

    result = _repository(session).get_latest_observed_at()

    assert result == cutoff
    assert "max(pedestrian_hourly_fact.observed_at)" in _compile(session.statements[0])
    assert session.closed

def test_get_latest_observed_at_returns_none_for_empty_table() -> None:
    session = FakeSession([])

    assert _repository(session).get_latest_observed_at() is None
    assert session.closed
~~~

Add get_latest_observed_at to both existing failure parametrizations. Assert a
session-factory failure and a scalar() query failure each raise
DataStoreUnavailableError and still close an entered session.

- [ ] **Step 2: Run the focused test to verify RED**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_postgres_repositories.py -q
~~~

Expected: the new tests fail because get_latest_observed_at does not exist.

- [ ] **Step 3: Implement the minimum adapter method**

Import func and add this method without changing existing query behavior:

~~~python
def get_latest_observed_at(self) -> datetime | None:
    statement = select(func.max(PedestrianHourlyFact.observed_at))
    try:
        with self._session_factory() as session:
            return session.scalar(statement)
    except SQLAlchemyError as exc:
        raise DataStoreUnavailableError("sensor data is unavailable") from exc
~~~

Keep the raw database return value intact. Timestamp validation belongs to
RuntimeHealthService, not the adapter.

- [ ] **Step 4: Run focused verification**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_postgres_repositories.py -q
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/api/postgres.py tests/unit/api/test_postgres_repositories.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/api/postgres.py tests/unit/api/test_postgres_repositories.py
~~~

Expected: all repository tests pass and Ruff reports no changes.

- [ ] **Step 5: Commit the adapter seam**

~~~powershell
git add src/urbanflow/api/postgres.py tests/unit/api/test_postgres_repositories.py
git commit -m "feat(api): expose latest observation readiness"
~~~

---

### Task 2: Implement deterministic runtime-health semantics

**Files:**

- Modify: src/urbanflow/api/services.py
- Modify: tests/unit/api/test_health.py

**Interfaces:**

~~~python
API_MAX_DATA_AGE_HOURS_ENV_VAR = "URBANFLOW_API_MAX_DATA_AGE_HOURS"

class ApiRuntimeConfigError(ValueError): ...

class DataReadinessRepository(Protocol):
    def get_latest_observed_at(self) -> datetime | None: ...

def parse_max_data_age(
    environ: Mapping[str, str],
) -> timedelta | None: ...

@dataclass(frozen=True, slots=True)
class RuntimeHealthService:
    data_readiness_repository: DataReadinessRepository | None
    model_provider_status: ComponentStatus
    model_version: str | None
    max_data_age: timedelta | None
    now: Callable[[], datetime] = utc_now

    def __call__(self) -> HealthResult: ...
~~~

The type annotation may use a local Literal-compatible alias if importing the
Pydantic type alias is inconvenient, but only available, unconfigured, and
unavailable are valid component values.

- [ ] **Step 1: Write failing data-age and direct health-service tests**

Add a small fake readiness repository and fixed UTC clock:

~~~python
class FakeReadinessRepository:
    def __init__(self, cutoff: datetime | None = None, error: Exception | None = None) -> None:
        self.cutoff = cutoff
        self.error = error

    def get_latest_observed_at(self) -> datetime | None:
        if self.error is not None:
            raise self.error
        return self.cutoff

def fixed_now() -> datetime:
    return datetime(2026, 7, 30, 12, tzinfo=UTC)
~~~

Add parameterized parse_max_data_age assertions for absent, blank, "1", "24",
"0", "-1", "1.5", and "not-a-number". Add RuntimeHealthService tests for:

1. no repository produces degraded, unconfigured data components, and no
   cutoff;
2. an aware cutoff within a two-hour threshold plus available model produces
   ok and preserves the cutoff/version;
3. an aware stale cutoff produces degraded with unavailable freshness and the
   real aware cutoff;
4. an aware cutoff with no threshold produces degraded with unconfigured
   freshness and the real cutoff;
5. no cutoff produces degraded, available data_store, unavailable freshness,
   and no cutoff;
6. DataStoreUnavailableError produces unavailable with both data components
   unavailable;
7. a naive cutoff or a cutoff after fixed_now produces degraded, unavailable
   freshness, and no returned data_cutoff_at;
8. model_provider_status unavailable does not make a readable data store return
   HTTP 503 by itself.

Keep the existing route tests for explicitly injected ok and unavailable
HealthResult values.

- [ ] **Step 2: Run the focused test to verify RED**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_health.py -q
~~~

Expected: collection or assertions fail because the runtime health types and
parser do not exist.

- [ ] **Step 3: Implement parsing and RuntimeHealthService**

Implement the parser entirely from the passed Mapping; do not read os.environ:

~~~python
def parse_max_data_age(environ: Mapping[str, str]) -> timedelta | None:
    raw_value = environ.get(API_MAX_DATA_AGE_HOURS_ENV_VAR)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        hours = int(raw_value)
    except ValueError as exc:
        raise ApiRuntimeConfigError(
            f"Invalid {API_MAX_DATA_AGE_HOURS_ENV_VAR} configuration."
        ) from exc
    if hours <= 0 or raw_value.strip() != str(hours):
        raise ApiRuntimeConfigError(
            f"Invalid {API_MAX_DATA_AGE_HOURS_ENV_VAR} configuration."
        )
    return timedelta(hours=hours)
~~~

Normalize a valid cutoff and clock to UTC only after verifying both are
offset-aware. A naive or future cutoff is not returned. Catch only
DataStoreUnavailableError from the repository. Construct HealthComponents and
HealthResult with:

- api_process always available when this callable runs;
- no repository: data_store and data_freshness unconfigured;
- repository exception: both data components unavailable and overall
  unavailable;
- readable store with None cutoff: store available and freshness unavailable;
- readable aware cutoff with no age threshold: store available and freshness
  unconfigured;
- readable aware cutoff within threshold: store and freshness available;
- stale, future, or naive cutoff: store available and freshness unavailable;
- overall ok only when model, data store, and freshness are available;
- otherwise degraded unless the data-store probe failed.

Use the existing API service name, package version, and UTC timestamp behavior.
Do not modify default_health or the route.

- [ ] **Step 4: Run focused verification**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_health.py -q
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/api/services.py tests/unit/api/test_health.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/api/services.py tests/unit/api/test_health.py
~~~

Expected: direct service tests and the existing HTTP response contract pass.

- [ ] **Step 5: Commit runtime health**

~~~powershell
git add src/urbanflow/api/services.py tests/unit/api/test_health.py
git commit -m "feat(api): report runtime readiness health"
~~~

---

### Task 3: Wire app configuration and update readiness wording

**Files:**

- Modify: src/urbanflow/api/app.py
- Modify: tests/unit/api/test_app.py
- Modify: src/urbanflow/dashboard/pages/today.py
- Modify: tests/unit/dashboard/test_today.py

**Interfaces:**

~~~python
def create_default_services(
    *,
    environ: Mapping[str, str] | None = None,
) -> ApiServices: ...
~~~

The function uses its resolved values mapping for database URL, artifact path,
and max-data-age parsing. It passes a RuntimeHealthService to ApiServices.

- [ ] **Step 1: Write failing factory and Dashboard wording tests**

Extend the app tests with fake artifacts containing:

~~~python
artifact = SimpleNamespace(
    manifest=SimpleNamespace(model_version="lightgbm-smoke-v1")
)
~~~

Add tests that assert:

1. no database URL still opens no session, reads no artifact, and exposes a
   RuntimeHealthService whose data repository is None;
2. database-only configuration creates no session during construction and
   produces a health service with data readiness but model unconfigured;
3. database plus a valid artifact makes the health service model available and
   exposes the manifest model version;
4. database plus invalid artifact leaves repository reads usable, model provider
   None, and health model component unavailable;
5. a valid age value is read from the supplied environ mapping even when
   os.environ has a different value;
6. malformed age values raise ApiRuntimeConfigError before returning services.

Update the existing fake session to support the max-observation scalar result
where a health request requires it. Retain lazy-construction assertions before
the request.

In the Today page test, assert degraded copy says the API reported degraded
configuration or availability, asks the user to review component statuses, and
does not claim that a healthy-looking service guarantees a selected sensor has
the 168-hour input and holiday coverage required for a forecast. Do not retain
the prior claim that degraded cannot indicate readiness.

- [ ] **Step 2: Run the focused tests to verify RED**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_app.py tests/unit/dashboard/test_today.py -q
~~~

Expected: factory tests fail because the default factory still uses
default_health and the old Dashboard wording is visible.

- [ ] **Step 3: Implement factory tri-state wiring and wording**

In app.py, determine artifact state without swallowing its meaning:

~~~python
model_provider = None
model_provider_status = "unconfigured"
model_version = None
if configured_artifact_path is not None and configured_artifact_path.strip():
    try:
        artifact = load_lightgbm_artifact(configured_artifact_path.strip())
    except LightGBMArtifactError:
        model_provider_status = "unavailable"
    else:
        model_provider = ArtifactBackedLightGBMForecastProvider(
            artifact=artifact,
            history_repository=repository,
        )
        model_provider_status = "available"
        model_version = artifact.manifest.model_version
~~~

Parse the age from the same values mapping before the no-database early-return
decision, so an invalid age is never silently accepted merely because the
database is absent. When database configuration is absent, continue to skip
artifact loading and use model_provider_status unconfigured. Pass repository,
model state, version, and age to
RuntimeHealthService. Preserve explicit services injection exactly as it is.

Change only the degraded warning sentence in today.py so it describes returned
degraded configuration or availability, directs the user to component statuses,
and does not promise per-sensor forecast readiness. Do not alter page layout,
requests, or controls.

- [ ] **Step 4: Run focused verification**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_app.py tests/unit/api/test_health.py tests/unit/dashboard/test_today.py -q
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/api/app.py src/urbanflow/dashboard/pages/today.py tests/unit/api/test_app.py tests/unit/dashboard/test_today.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/api/app.py src/urbanflow/dashboard/pages/today.py tests/unit/api/test_app.py tests/unit/dashboard/test_today.py
~~~

Expected: default no-I/O behavior, configured health semantics, and Dashboard
wording all pass.

- [ ] **Step 5: Commit factory and wording**

~~~powershell
git add src/urbanflow/api/app.py tests/unit/api/test_app.py src/urbanflow/dashboard/pages/today.py tests/unit/dashboard/test_today.py
git commit -m "feat(api): wire configured readiness health"
~~~

---

### Task 4: Add a bounded opt-in configured-serving smoke

**Files:**

- Create: src/urbanflow/api/serving_e2e_smoke.py
- Create: scripts/smoke_test_serving_e2e.py
- Create: tests/unit/api/test_serving_e2e_smoke.py

**Interfaces:**

~~~python
SMOKE_DATABASE_URL_ENV_VAR = "URBANFLOW_SMOKE_DATABASE_URL"

@dataclass(frozen=True)
class ServingE2ESmokeResult:
    schema_name: str
    location_id: int
    health_status: str
    data_cutoff_at: str
    model_version: str
    history_count: int
    forecast_horizons: list[int]

def run_serving_e2e_smoke(
    database_url: str,
    *,
    schema_name: str | None = None,
) -> ServingE2ESmokeResult: ...

def build_parser() -> argparse.ArgumentParser: ...
def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int: ...
~~~

- [ ] **Step 1: Write offline failing smoke tests**

Create tests that never open a PostgreSQL connection or start a real process.
Cover:

1. safe generated schema names and dangerous identifier rejection by reusing
   postgres_smoke.validate_smoke_schema_name;
2. missing --database-url and absent smoke environment returns exit code 2 and
   names URBANFLOW_SMOKE_DATABASE_URL;
3. the child database URL preserves the base URL but adds a URL-encoded
   libpq options value for only the validated search path;
4. child environment begins as os.environ.copy(), removes every inherited
   URBANFLOW_ variable, then contains only the smoke database URL, temporary
   artifact path, and a positive max-data-age value among UrbanFlow settings;
5. result dataclass serializes safely without URL or artifact fields;
6. startup timeout and child early exit include only a bounded log tail and
   raise a smoke error;
7. cleanup invokes terminate, then wait with a fixed deadline, then kill and
   wait only when terminate did not stop the fake process;
8. cleanup still closes the fake DashboardApiClient, closes the temporary log,
   drops only a created schema, and disposes the engine when HTTP assertion
   code raises.

Inject a process factory, clock, sleep function, free-port allocator, HTTP
poller, and DashboardApiClient factory into private helpers. Keep
run_serving_e2e_smoke as the public production entry point.

- [ ] **Step 2: Run the focused test to verify RED**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_serving_e2e_smoke.py -q
~~~

Expected: collection fails because serving_e2e_smoke and its script do not
exist.

- [ ] **Step 3: Implement the isolated smoke lifecycle**

Use existing adapter and artifact primitives rather than new production
interfaces:

~~~python
schema = postgres_smoke.validate_smoke_schema_name(
    schema_name or f"urbanflow_serving_e2e_{uuid4().hex[:12]}"
)
engine = create_database_engine(database_url)
with engine.begin() as connection:
    connection.exec_driver_sql(f"CREATE SCHEMA {_quote_identifier(schema)}")
    connection.exec_driver_sql(f"SET search_path TO {_quote_identifier(schema)}")
    Base.metadata.create_all(connection)
~~~

Seed one active sensor and 192 contiguous Melbourne-local observations ending
within one hour of a UTC clock. Build the temporary supervised frame, holiday
calendar, and artifact with the existing LightGBM smoke conventions, but use a
two-hour child age threshold to tolerate bounded startup time.

Build a child-only schema URL with sqlalchemy.engine.make_url and an options
query value of -csearch_path=<validated schema>. Preserve any pre-existing URL
query items, render it for the child with
render_as_string(hide_password=False), and never log that rendered value.
Start:

~~~python
command = [
    sys.executable,
    "-m",
    "uvicorn",
    "urbanflow.api.app:app",
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
]
~~~

Do not add reload, workers, a shell, or a fixed port. Redirect stdout and
stderr to a temporary log file rather than unread PIPEs. Poll the loopback
health endpoint with time.monotonic() deadlines and short HTTP timeouts. Once
ready, use a real DashboardApiClient to assert:

~~~python
health = client.get_health()
sensors = client.list_sensors(active_only=True)
history = client.get_history(location_id, start=history_start, end=history_end)
forecast = client.get_forecast(location_id, horizon=24)
~~~

Require health.status == "ok", every component available, a matching seeded
cutoff/model version, exactly the seeded active sensor, ordered returned
history, and forecast horizons 1 through 24. Include health_status and
history_count, but no URL or path, in the successful result. DashboardApiClient already
validates timestamp awareness, model response shape, finite predictions, and
non-negative values.

In one finally block, close the client, terminate then wait for at most ten
seconds, kill only if still alive, close the log, drop the validated schema
only if schema_created is true, dispose the engine, and let
TemporaryDirectory remove artifacts. Truncate any failure log tail and never
include connection strings or paths in it.

- [ ] **Step 4: Add the executable wrapper and CLI behavior**

Make scripts/smoke_test_serving_e2e.py only:

~~~python
from urbanflow.api.serving_e2e_smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
~~~

In main, accept --database-url and --schema-name, use only the explicit value
or URBANFLOW_SMOKE_DATABASE_URL, return 2 for configuration/schema errors,
return 1 for smoke, SQLAlchemy, artifact, HTTP, or process failures, and print
only json.dumps(asdict(result), sort_keys=True) on success.

- [ ] **Step 5: Run focused verification**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_serving_e2e_smoke.py tests/unit/api/test_lightgbm_forecast_smoke.py tests/unit/api/test_postgres_smoke.py -q
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/api/serving_e2e_smoke.py scripts/smoke_test_serving_e2e.py tests/unit/api/test_serving_e2e_smoke.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/api/serving_e2e_smoke.py scripts/smoke_test_serving_e2e.py tests/unit/api/test_serving_e2e_smoke.py
~~~

Expected: all new tests are offline, current smoke regressions pass, and no
new process is left running.

- [ ] **Step 6: Commit the configured-serving smoke**

~~~powershell
git add src/urbanflow/api/serving_e2e_smoke.py scripts/smoke_test_serving_e2e.py tests/unit/api/test_serving_e2e_smoke.py
git commit -m "test(api): add configured serving smoke"
~~~

---

### Task 5: Document the operator path and perform final verification

**Files:**

- Modify: README.md
- Review: all files changed in main...HEAD

- [ ] **Step 1: Write README acceptance assertions and update the operator guide**

Add one focused documentation-oriented assertion only if existing tests already
validate README command text; otherwise review the rendered Markdown manually.
Update the FastAPI section with:

1. a compact Health matrix for no database, readable database without an age
   threshold, readable fresh database with a valid artifact, stale/empty data,
   unreadable database, and invalid artifact;
2. the meaning and validation rule of URBANFLOW_API_MAX_DATA_AGE_HOURS;
3. the fact that data_cutoff_at is the latest database observation, never the
   model training cutoff, and health ok does not guarantee every sensor has
   168 contiguous forecast input rows;
4. an explicit manual serving-smoke command:

~~~powershell
$env:URBANFLOW_SMOKE_DATABASE_URL = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"
python scripts/smoke_test_serving_e2e.py
~~~

Explain that it creates and removes an isolated schema and a temporary local
artifact, starts only temporary Uvicorn, makes no network request, does not
start Streamlit, does not run in CI, and never uses the normal database URL as
a fallback.

- [ ] **Step 2: Review scope and run focused documentation checks**

Run:

~~~powershell
git diff --check main...HEAD
rg -n "URBANFLOW_API_MAX_DATA_AGE_HOURS|smoke_test_serving_e2e|data_cutoff_at" README.md
git diff -- main...HEAD -- README.md src/urbanflow/api app tests
~~~

Expected: documentation covers the new setting and smoke, and the diff contains
no Docker, migration, model-training, monitoring, or new Dashboard-page work.

- [ ] **Step 3: Run the full automated quality gate**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest -q
git status --short --branch
~~~

Expected: all tests pass, formatting is clean, and no unintended file is
modified. Existing deferred warnings may remain but must not increase.

- [ ] **Step 4: Run the manual configured-serving smoke only when explicitly configured**

If URBANFLOW_SMOKE_DATABASE_URL is available and points to the local test
database, run:

~~~powershell
python scripts/smoke_test_serving_e2e.py
~~~

Expected: JSON with a temporary schema name, smoke location ID, real cutoff,
artifact-derived model version, and horizons 1 through 24. Confirm that no
Uvicorn process remains and the temporary schema is gone. If the variable is
absent, record the manual smoke as not run; do not invent a URL or start a
database.

- [ ] **Step 5: Commit the documentation and verification-ready changes**

~~~powershell
git add README.md
git commit -m "docs: explain runtime readiness checks"
~~~

- [ ] **Step 6: Conduct branch review and handoff**

Review the branch against the approved specification:

~~~powershell
git diff --check main...HEAD
git diff --stat main...HEAD
git diff main...HEAD -- src/urbanflow/api src/urbanflow/dashboard README.md tests scripts
git status --short --branch
~~~

Confirm all five public routes and public schema shapes are unchanged, no
connection starts during default app construction, no manual smoke is in CI,
and no background process survives its test path. After review, merge only the
verified local branch into main, rerun the full quality gate on main, push only
main, and remove the worktree according to docs/development_workflow.md.
