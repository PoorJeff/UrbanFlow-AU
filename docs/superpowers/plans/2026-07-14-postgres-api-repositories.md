# PostgreSQL API Repositories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> \`superpowers:subagent-driven-development\` or
> \`superpowers:executing-plans\` to implement this plan task-by-task. Steps use
> checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Wire the existing sensor and history API contracts to explicitly
configured PostgreSQL reads while preserving safe default behavior and keeping
model serving out of scope.

**Architecture:** Add a small SQLAlchemy adapter under \`urbanflow.api\` that
implements the existing repository protocols and maps ORM rows into the API
records. Extend the app factory with an environment-aware default-services
builder; an absent URL keeps empty repositories, while a configured URL creates
an Engine without opening a database connection. Add database-free unit tests,
an opt-in temporary-schema PostgreSQL smoke, and truthful documentation.

**Tech Stack:** Python 3.11+, SQLAlchemy 2, psycopg 3, FastAPI 0.139.x,
Pydantic 2, pytest, httpx ASGI transport, Uvicorn 0.51.x, Ruff.

## Global Constraints

- Keep business paths under \`/api/v1\` and \`/health\` unversioned.
- Add no runtime dependency; SQLAlchemy, psycopg, FastAPI, pytest, and Ruff
  already exist in \`pyproject.toml\`.
- With no \`URBANFLOW_DATABASE_URL\`, startup must not create an Engine, connect
  to PostgreSQL, contact Melbourne Open Data, load a model, or require MLflow.
- A whitespace-only database URL behaves like an absent URL.
- A malformed or unsupported non-empty URL raises \`DatabaseConfigError\` during
  default service construction. A valid but unreachable URL remains lazy and
  produces \`503 data_store_unavailable\` only when a read fails.
- \`active_only=true\` means exactly \`SensorDim.status == "A"\`; do not accept
  a text heuristic such as \`"active"\`.
- History reads use timezone-aware \`[start, end)\` bounds and
  \`ORDER BY observed_at ASC\`. Preserve an aware value's instant, not its
  original zone-object representation.
- Repository read failures from session creation or statement execution become
  \`DataStoreUnavailableError\`, which existing API code renders as
  \`503 data_store_unavailable\`.
- Do not alter models, migrations, ingestion, upserts, model artifacts,
  \`ForecastModelProvider\`, forecast behavior, health probing, Dashboard,
  Evidently, Docker Compose, or Melbourne API request behavior.
- Routine pytest remains PostgreSQL- and network-free. Existing full-suite
  pandas/NumPy deprecation warnings are out of scope and must not increase.
- Follow \`docs/development_workflow.md\`: commit each reviewed task with a
  Conventional Commit, run the full gate before merge, and push only \`main\`.

---

### Task 1: Add the read-only PostgreSQL repository adapter

**Files:**

- Create: \`src/urbanflow/api/postgres.py\`
- Create: \`tests/unit/api/test_postgres_repositories.py\`

**Interfaces:**

- Consumes: \`SensorDim\`, \`PedestrianHourlyFact\`,
  \`sqlalchemy.orm.sessionmaker[Session]\`, \`SensorRecord\`,
  \`HistoryRecord\`, and \`DataStoreUnavailableError\`.
- Produces:

  ~~~python
  class PostgresSensorHistoryRepository:
      def __init__(self, session_factory: sessionmaker[Session]) -> None: ...
      def list_sensors(self, active_only: bool) -> list[SensorRecord]: ...
      def get_sensor(self, location_id: int) -> SensorRecord | None: ...
      def get_history(
          self,
          location_id: int,
          start: datetime,
          end: datetime,
      ) -> list[HistoryRecord]: ...
  ~~~

- Later tasks inject one instance into both \`ApiServices.sensor_repository\`
  and \`ApiServices.history_repository\`.

- [x] **Step 1: Write focused failing adapter tests**

  Create test-only controlled session helpers that record the \`Select\`
  statement and return prepared ORM objects without a database. Import \`UTC\`
  from \`datetime\` for instant comparisons:

  ~~~python
  class FakeScalarResult:
      def __init__(
          self,
          rows: list[object],
          result_error: SQLAlchemyError | None = None,
      ) -> None:
          self.rows = rows
          self.result_error = result_error

      def all(self) -> list[object]:
          if self.result_error is not None:
              raise self.result_error
          return list(self.rows)

      def one_or_none(self) -> object | None:
          if self.result_error is not None:
              raise self.result_error
          if len(self.rows) > 1:
              raise AssertionError("expected at most one row")
          return self.rows[0] if self.rows else None

  class FakeSession:
      def __init__(
          self,
          rows: list[object],
          scalars_error: SQLAlchemyError | None = None,
          result_error: SQLAlchemyError | None = None,
      ) -> None:
          self.rows = rows
          self.scalars_error = scalars_error
          self.result_error = result_error
          self.statements = []

      def __enter__(self) -> FakeSession:
          return self

      def __exit__(self, exc_type, exc, traceback) -> bool:
          return False

      def scalars(self, statement):
          self.statements.append(statement)
          if self.scalars_error is not None:
              raise self.scalars_error
          return FakeScalarResult(self.rows, self.result_error)
  ~~~

  Add tests that:

  1. provide two \`SensorDim\` rows with \`A\` and \`I\` statuses and assert the
     mapped \`SensorRecord\` fields;
  2. compile the recorded active query with
     \`postgresql.dialect()\` and
     \`compile_kwargs={"literal_binds": True}\`, then assert it includes
     \`sensor_dim.status = 'A'\` and \`ORDER BY sensor_dim.location_id\`;
  3. compile the non-active query and assert it has no status predicate;
  4. cover \`get_sensor\` for one row and no row;
  5. provide deliberately unordered \`PedestrianHourlyFact\` rows with an aware
     \`source_observed_at\`; assert the mapped record is timezone-aware and
     \`record.observed_at.astimezone(UTC) == source_observed_at.astimezone(UTC)\`,
     then compile the
    history query to assert the exact location and \`>= start\`, \`< end\`,
    and ascending-order predicates;
  6. parameterize a session-factory failure and a \`scalars()\` failure for
     each public repository method, asserting \`DataStoreUnavailableError\`;
  7. make \`FakeScalarResult.all()\` raise \`SQLAlchemyError\` for both
     \`list_sensors\` and \`get_history\`, and make
     \`FakeScalarResult.one_or_none()\` raise it for \`get_sensor\`; assert
     every row-consumption failure is also translated to
     \`DataStoreUnavailableError\`.

- [x] **Step 2: Run the focused tests and confirm RED**

  Run:

  ~~~powershell
  & .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_postgres_repositories.py -q
  ~~~

  Expected: collection fails because \`urbanflow.api.postgres\` and
  \`PostgresSensorHistoryRepository\` do not exist.

- [x] **Step 3: Implement the smallest adapter**

  Create \`src/urbanflow/api/postgres.py\` with a single active-status constant,
  small mapping helpers, and one error wrapper:

  ~~~python
  ACTIVE_SENSOR_STATUS = "A"

  class PostgresSensorHistoryRepository:
      def __init__(self, session_factory: sessionmaker[Session]) -> None:
          self._session_factory = session_factory

      def list_sensors(self, active_only: bool) -> list[SensorRecord]:
          statement = select(SensorDim).order_by(SensorDim.location_id)
          if active_only:
              statement = statement.where(SensorDim.status == ACTIVE_SENSOR_STATUS)
          try:
              with self._session_factory() as session:
                  rows = session.scalars(statement).all()
          except SQLAlchemyError as exc:
              raise DataStoreUnavailableError("could not read sensors") from exc
          return [_to_sensor_record(row) for row in rows]

      def get_sensor(self, location_id: int) -> SensorRecord | None:
          statement = select(SensorDim).where(SensorDim.location_id == location_id)
          try:
              with self._session_factory() as session:
                  row = session.scalars(statement).one_or_none()
          except SQLAlchemyError as exc:
              raise DataStoreUnavailableError("could not read sensor") from exc
          return None if row is None else _to_sensor_record(row)

      def get_history(
          self,
          location_id: int,
          start: datetime,
          end: datetime,
      ) -> list[HistoryRecord]:
          statement = (
              select(PedestrianHourlyFact)
              .where(
                  PedestrianHourlyFact.location_id == location_id,
                  PedestrianHourlyFact.observed_at >= start,
                  PedestrianHourlyFact.observed_at < end,
              )
              .order_by(PedestrianHourlyFact.observed_at)
          )
          try:
              with self._session_factory() as session:
                  rows = session.scalars(statement).all()
          except SQLAlchemyError as exc:
              raise DataStoreUnavailableError("could not read sensor history") from exc
          return [_to_history_record(row) for row in rows]
  ~~~

  Implement \`_to_sensor_record(SensorDim) -> SensorRecord\` and
  \`_to_history_record(PedestrianHourlyFact) -> HistoryRecord\` as direct
  field mappings. Do not catch non-SQLAlchemy programming errors and do not
  add router or FastAPI imports.

- [x] **Step 4: Run focused verification**

  Run:

  ~~~powershell
  & .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_postgres_repositories.py -q
  & .\.venv\Scripts\python.exe -m ruff check src/urbanflow/api/postgres.py tests/unit/api/test_postgres_repositories.py
  & .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/api/postgres.py tests/unit/api/test_postgres_repositories.py
  ~~~

  Expected: all new repository tests pass; Ruff reports no findings and no
  formatting changes.

- [x] **Step 5: Commit the adapter task**

  ~~~powershell
  git add src/urbanflow/api/postgres.py tests/unit/api/test_postgres_repositories.py
  git commit -m "feat(api): add postgres sensor history repository"
  ~~~

### Task 2: Wire explicit database configuration into the default app services

**Files:**

- Modify: \`src/urbanflow/api/app.py\`
- Create: \`tests/conftest.py\`
- Modify: \`tests/unit/api/test_app.py\`

**Interfaces:**

- Consumes: \`DATABASE_URL_ENV_VAR\`, \`DatabaseConfigError\`,
  \`create_database_engine\`, \`create_session_factory\`, and
  \`PostgresSensorHistoryRepository\`.
- Produces:

  ~~~python
  def create_default_services(
      *, environ: Mapping[str, str] | None = None
  ) -> ApiServices: ...
  ~~~

- \`create_app(*, services: ApiServices | None = None)\` continues to be the
  public app factory. Explicit \`services\` always bypasses
  \`create_default_services\`.

- [x] **Step 1: Add failing default-wiring tests and collection-time isolation**

  Add \`tests/conftest.py\`:

  ~~~python
  import os

  from urbanflow.database.config import DATABASE_URL_ENV_VAR

  os.environ.pop(DATABASE_URL_ENV_VAR, None)
  ~~~

  This runs before test modules import the module-level Uvicorn \`app\`.

  In \`test_app.py\`, add tests that:

  1. call \`create_default_services(environ={})\` and assert both repositories
     retain the existing empty defaults;
  2. call it with only whitespace and assert the same result;
  3. monkeypatch \`create_database_engine\` and
     \`create_session_factory\`, pass a valid URL, and assert one
     \`PostgresSensorHistoryRepository\` instance is used for both repository
     slots without invoking a session factory;
  4. add a \`StatementAwareSession\` test helper that returns controlled \`A\`
     and \`I\` \`SensorDim\` rows plus deliberately unordered aware fact rows
     according to the selected ORM entity and the active-status predicate.
     Set a valid URL with \`monkeypatch.setenv\`, monkeypatch the engine and
     session-factory constructors, then call the default \`create_app()\` and
     assert successful HTTP reads through both endpoints:

     ~~~python
     application = app_module.create_app()

     sensors = api_get(
         application,
         "/api/v1/sensors",
         params={"active_only": "true"},
     )
     history = api_get(
         application,
         "/api/v1/sensors/999001/history",
         params={
             "start": "2026-01-01T00:00:00+00:00",
             "end": "2026-01-02T00:00:00+00:00",
         },
     )

     assert sensors.status_code == 200
     assert [row["location_id"] for row in sensors.json()["data"]] == [999001]
     assert history.status_code == 200
     assert [row["pedestrian_count"] for row in history.json()["data"]] == [7, 42]
     ~~~

     This is database-free: it proves the default app wiring reaches the real
     adapter and returns the configured repository's data without opening a
     network connection;
  5. pass \`not-a-sqlalchemy-url\` and assert
    \`DatabaseConfigError\` rather than empty fallback;
  6. pass a syntactically valid PostgreSQL URL while monkeypatching the session
    factory to raise \`OperationalError\` on use; assert app construction
    succeeds and \`GET /api/v1/sensors\` returns \`503
    data_store_unavailable\`;
  7. inject a sentinel \`ApiServices\` into \`create_app\` while a database URL
    is configured, and assert \`application.state.services is sentinel\`.

- [x] **Step 2: Run the focused tests and confirm RED**

  Run:

  ~~~powershell
  & .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_app.py -q
  ~~~

  Expected: import or attribute failure for \`create_default_services\`.

- [x] **Step 3: Implement lazy default service construction**

  In \`app.py\`, add:

  ~~~python
  def create_default_services(
      *, environ: Mapping[str, str] | None = None
  ) -> ApiServices:
      values = os.environ if environ is None else environ
      configured_url = values.get(DATABASE_URL_ENV_VAR)
      if configured_url is None or not configured_url.strip():
          return ApiServices()
      try:
          engine = create_database_engine(configured_url.strip())
      except ArgumentError as exc:
          raise DatabaseConfigError(
              f"Invalid {DATABASE_URL_ENV_VAR} configuration."
          ) from exc
      session_factory = create_session_factory(engine)
      repository = PostgresSensorHistoryRepository(session_factory)
      return ApiServices(
          sensor_repository=repository,
          history_repository=repository,
      )
  ~~~

  Change the existing \`create_app\` assignment to:

  ~~~python
  application.state.services = (
      services if services is not None else create_default_services()
  )
  ~~~

  Import only \`os\`, \`Mapping\`, \`ArgumentError\`, existing database helpers,
  and the new adapter. Do not perform \`engine.connect()\`, add a health probe,
  or create a model provider.

- [x] **Step 4: Run focused verification**

  Run:

  ~~~powershell
  & .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_app.py tests/unit/api/test_sensors.py -q
  & .\.venv\Scripts\python.exe -m ruff check src/urbanflow/api/app.py tests/conftest.py tests/unit/api/test_app.py
  & .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/api/app.py tests/conftest.py tests/unit/api/test_app.py
  ~~~

  Expected: configured and unconfigured wiring paths pass without a network or
  PostgreSQL server.

- [x] **Step 5: Commit the wiring task**

  ~~~powershell
  git add src/urbanflow/api/app.py tests/conftest.py tests/unit/api/test_app.py
  git commit -m "feat(api): wire optional postgres repositories"
  ~~~

### Task 3: Lock down the existing HTTP unavailable-data paths

**Files:**

- Modify: \`tests/unit/api/test_history.py\`
- Modify: \`tests/unit/api/test_sensors.py\` only if an adapter-backed list
  failure test is needed beyond the existing route test

**Interfaces:**

- Consumes: existing \`ApiServices\`, \`HistoryService\`,
  \`DataStoreUnavailableError\`, and the error response contract.
- Produces complete HTTP coverage for list reads, sensor lookup reads, and
  history-query reads when storage is unavailable.

- [x] **Step 1: Add the missing history sensor-lookup failure test**

  Define a minimal test-only repository:

  ~~~python
  class FailingLookupSensorRepository:
      def list_sensors(self, active_only: bool) -> list[SensorRecord]:
          return []

      def get_sensor(self, location_id: int) -> SensorRecord | None:
          raise DataStoreUnavailableError("sensor lookup is unavailable")
  ~~~

  Inject it as \`sensor_repository\` with an inert history repository, request
  a valid history range, and assert the exact standard \`503\` error body.

  Keep the existing catalog list-failure test and history-query-failure test.
  Together, the three tests prove the router/service paths that the new
  PostgreSQL adapter can trigger.

- [x] **Step 2: Run the focused test and record the existing contract**

  Run:

  ~~~powershell
  & .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_history.py -q
  ~~~

  Expected: the new lookup-failure assertion passes without a production code
  change because \`HistoryService._ensure_sensor_exists\` already maps this
  error. This is a characterization test that protects a pre-existing route
  behavior newly exercised by the PostgreSQL adapter.

- [x] **Step 3: Keep production code unchanged**

  Do not modify \`src/urbanflow/api/services.py\` or router production code in
  this task. The purpose is to document the existing lookup-failure behavior
  before the PostgreSQL adapter is used at runtime.

- [x] **Step 4: Run focused verification**

  Run:

  ~~~powershell
  & .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_sensors.py tests/unit/api/test_history.py -q
  & .\.venv\Scripts\python.exe -m ruff check tests/unit/api/test_sensors.py tests/unit/api/test_history.py
  & .\.venv\Scripts\python.exe -m ruff format --check tests/unit/api/test_sensors.py tests/unit/api/test_history.py
  ~~~

  Expected: all storage-unavailable HTTP paths return the same project error
  response; focused tests add no warnings.

- [x] **Step 5: Commit the HTTP coverage task**

  ~~~powershell
  git add tests/unit/api/test_sensors.py tests/unit/api/test_history.py
  git commit -m "test(api): cover postgres repository failures"
  ~~~

### Task 4: Add an opt-in temporary-schema PostgreSQL adapter smoke

**Files:**

- Create: \`src/urbanflow/api/postgres_smoke.py\`
- Create: \`scripts/smoke_test_postgres_api.py\`
- Create: \`tests/unit/api/test_postgres_smoke.py\`

**Interfaces:**

- Consumes: \`URBANFLOW_SMOKE_DATABASE_URL\`,
  \`create_database_engine\`, \`Base\`, the existing upsert helpers, and
  \`PostgresSensorHistoryRepository\`.
- Produces:

  ~~~python
  @dataclass(frozen=True)
  class PostgresApiRepositorySmokeResult:
      schema_name: str
      all_sensor_location_ids: list[int]
      active_sensor_location_ids: list[int]
      history_count: int

  def run_postgres_api_repository_smoke(
      database_url: str, *, schema_name: str | None = None
  ) -> PostgresApiRepositorySmokeResult: ...

  def main(
      argv: list[str] | None = None, *, environ: Mapping[str, str] | None = None
  ) -> int: ...
  ~~~

- [x] **Step 1: Write failing unit tests for the smoke CLI and safety**

  Mirror the project’s \`tests/unit/database/test_smoke.py\` style:

  1. missing \`URBANFLOW_SMOKE_DATABASE_URL\` returns exit code \`2\`;
  2. an explicit URL and schema name call
     \`run_postgres_api_repository_smoke\` and print its JSON result;
  3. unsafe schema names are rejected before any engine is created;
  4. a fake engine that fails on \`CREATE SCHEMA\` is disposed and never drops a
     schema;
  5. the script \`--help\` exits successfully and names the API repository
     smoke.

- [x] **Step 2: Run the focused tests and confirm RED**

  Run:

  ~~~powershell
  & .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_postgres_smoke.py -q
  ~~~

  Expected: collection fails because \`urbanflow.api.postgres_smoke\` and the
  thin script do not exist.

- [x] **Step 3: Implement the isolated-schema smoke**

  Implement a safe lowercase schema-name validator and generated
  \`urbanflow_api_smoke_<12-hex>\` name in \`postgres_smoke.py\`. Import
  \`UTC\` from \`datetime\` and \`ZoneInfo\` from \`zoneinfo\` for the
  timestamp-instant assertion. Follow this exact lifecycle:

  1. create the Engine;
  2. create the temporary schema and mark it as created only after the command
     succeeds;
  3. inside one \`engine.begin()\` connection, run \`SET search_path TO
     "<schema>"\`, call \`Base.metadata.create_all(connection)\`, and build a
     \`sessionmaker(bind=connection, autoflush=False, expire_on_commit=False)\`;
  4. define \`source_observed_at\` as an aware non-UTC timestamp, then write
     an \`A\` sensor at location \`999001\`, an \`I\` sensor at \`999002\`,
     and one hourly row for \`999001\` through the existing upsert helpers;
  5. instantiate \`PostgresSensorHistoryRepository\` with that same
    connection-bound session factory; assert its all-sensor result is
    \`[999001, 999002]\`, its active result is \`[999001]\`, and its history
     result is one aware record with count \`42\`. Assert both that the
     returned timestamp is aware and that
     \`history[0].observed_at.astimezone(UTC) == source_observed_at.astimezone(UTC)\`;
  6. return the result dataclass;
  7. in \`finally\`, drop only the validated temporary schema and dispose the
    Engine.

  The CLI accepts \`--database-url\` and \`--schema-name\`, resolves the former
  from \`URBANFLOW_SMOKE_DATABASE_URL\` when omitted, returns \`2\` for invalid
  input, \`1\` for \`SQLAlchemyError\`, and \`0\` after printing sorted JSON.
  The script contains only:

  ~~~python
  from urbanflow.api.postgres_smoke import main

  if __name__ == "__main__":
      raise SystemExit(main())
  ~~~

- [x] **Step 4: Run focused verification**

  Run:

  ~~~powershell
  & .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_postgres_smoke.py -q
  & .\.venv\Scripts\python.exe -m ruff check src/urbanflow/api/postgres_smoke.py scripts/smoke_test_postgres_api.py tests/unit/api/test_postgres_smoke.py
  & .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/api/postgres_smoke.py scripts/smoke_test_postgres_api.py tests/unit/api/test_postgres_smoke.py
  ~~~

  If a local PostgreSQL URL is intentionally available, also run:

  ~~~powershell
  $env:URBANFLOW_SMOKE_DATABASE_URL = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"
  & .\.venv\Scripts\python.exe scripts/smoke_test_postgres_api.py
  ~~~

  Expected manual result: JSON with \`active_sensor_location_ids:[999001]\`,
  \`all_sensor_location_ids:[999001,999002]\`, and \`history_count:1\`.

- [x] **Step 5: Commit the smoke task**

  ~~~powershell
  git add src/urbanflow/api/postgres_smoke.py scripts/smoke_test_postgres_api.py tests/unit/api/test_postgres_smoke.py
  git commit -m "test(api): add postgres repository smoke"
  ~~~

### Task 5: Synchronize documentation and verify the completed slice

**Files:**

- Modify: \`README.md\`
- Modify: \`urbanflow-au_requirements.md\`
- Modify: \`docs/superpowers/plans/2026-07-14-postgres-api-repositories.md\`

**Interfaces:**

- Consumes: completed default app wiring, the read adapter, and the smoke CLI.
- Produces accurate project-status and local-run documentation without claiming
  a deployed forecast model.

- [x] **Step 1: Update the user-facing documentation**

  In the README FastAPI section:

  1. replace the claim that database-backed repositories remain future work;
  2. state that \`URBANFLOW_DATABASE_URL\` opt-in wiring serves persisted
     sensors and history, while no URL retains the empty/default behavior;
  3. retain the explicit \`503 model_unavailable\` description for forecast;
  4. add a PowerShell example that sets \`URBANFLOW_DATABASE_URL\`, starts
     Uvicorn, calls \`/api/v1/sensors?active_only=true\`, and calls bounded
     history;
  5. add \`scripts/smoke_test_postgres_api.py\` beside the existing manual
     persistence smoke command and state that it creates and drops an isolated
     schema.

  In section 10 of \`urbanflow-au_requirements.md\`, replace the statement that
  database reads are a future slice with a precise statement that sensor and
  history reads are available only when the PostgreSQL URL is explicitly
  configured. Keep model artifact loading and the real forecast provider marked
  as future work.

- [x] **Step 2: Mark plan tasks and run focused documentation checks**

  Change every completed task checkbox in this plan from \`[ ]\` to \`[x]\`.
  Run:

 ~~~powershell
  $ErrorActionPreference = "Stop"
 git diff --check
  if ($LASTEXITCODE -ne 0) { throw "git diff --check failed." }
 & .\.venv\Scripts\python.exe -m ruff check .
  if ($LASTEXITCODE -ne 0) { throw "Ruff lint failed." }
 & .\.venv\Scripts\python.exe -m ruff format --check .
  if ($LASTEXITCODE -ne 0) { throw "Ruff format check failed." }
 ~~~

  Expected: no whitespace errors, no Ruff findings, and every file already
  formatted.

- [x] **Step 3: Run the full verification gate**

  Run:

 ~~~powershell
  $ErrorActionPreference = "Stop"
 & .\.venv\Scripts\python.exe -m pytest -q
  if ($LASTEXITCODE -ne 0) { throw "pytest failed." }
 ~~~

  Expected: all existing and new tests pass. The known pandas/NumPy
  \`Timedelta\` deprecation warnings may remain; no new warning category is
  acceptable.

  Run this bounded Uvicorn smoke from the feature worktree. It preserves any
  process-local database URL, starts a hidden process, has a 20-second ready
  deadline, and always stops only the process it created:

  ~~~powershell
  $ErrorActionPreference = "Stop"
  $hadDatabaseUrl = Test-Path Env:URBANFLOW_DATABASE_URL
  $previousDatabaseUrl = $env:URBANFLOW_DATABASE_URL
  $stdout = New-TemporaryFile
  $stderr = New-TemporaryFile
  $process = $null
  try {
      Remove-Item Env:URBANFLOW_DATABASE_URL -ErrorAction SilentlyContinue
      $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
      $listener.Start()
      $port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
      $listener.Stop()
      $process = Start-Process `
          -FilePath .\.venv\Scripts\python.exe `
          -ArgumentList @("-m", "uvicorn", "urbanflow.api.app:app", "--host", "127.0.0.1", "--port", "$port") `
          -PassThru `
          -WindowStyle Hidden `
          -RedirectStandardOutput $stdout `
          -RedirectStandardError $stderr
      $deadline = (Get-Date).AddSeconds(20)
      $health = $null
      while ((Get-Date) -lt $deadline -and $null -eq $health) {
          try {
              $candidate = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2
              if ($candidate.StatusCode -eq 200) { $health = $candidate }
          } catch {}
          if ($null -eq $health) { Start-Sleep -Milliseconds 250 }
      }
      if ($null -eq $health) {
          throw "Uvicorn did not become healthy within 20 seconds: $(Get-Content -Raw $stderr)"
      }
      $openapi = Invoke-RestMethod -Uri "http://127.0.0.1:$port/openapi.json" -TimeoutSec 2
      $expectedPaths = @(
          "/health",
          "/api/v1/sensors",
          "/api/v1/sensors/{location_id}/history",
          "/api/v1/sensors/{location_id}/forecast",
          "/api/v1/model/metrics"
      )
      $actualPaths = @($openapi.paths.PSObject.Properties.Name)
      if (Compare-Object $expectedPaths $actualPaths) { throw "OpenAPI route set changed." }
      $sensorResponse = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/v1/sensors" -TimeoutSec 2
      if (@($sensorResponse.data).Count -ne 0) { throw "Unconfigured sensors endpoint was not empty." }
  } finally {
      if ($null -ne $process -and -not $process.HasExited) {
          Stop-Process -Id $process.Id -Force
      }
      Remove-Item $stdout, $stderr -ErrorAction SilentlyContinue
      if ($hadDatabaseUrl) {
          $env:URBANFLOW_DATABASE_URL = $previousDatabaseUrl
      } else {
          Remove-Item Env:URBANFLOW_DATABASE_URL -ErrorAction SilentlyContinue
      }
  }
  ~~~

  Expected: `/health` is `200`, OpenAPI contains exactly the five documented
  paths, and with no `URBANFLOW_DATABASE_URL`, `GET /api/v1/sensors` is `200`
  with an empty `data` list.

- [x] **Step 4: Commit the documentation and closeout**

  ~~~powershell
  git add README.md urbanflow-au_requirements.md docs/superpowers/plans/2026-07-14-postgres-api-repositories.md
  git commit -m "docs: describe postgres api repositories"
  git status --short --branch
  ~~~

  Expected: a clean feature worktree on \`codex/postgres-api-repositories\`.

### Task 6: Review, integrate, and verify main

**Files:**

- Verify only; do not add scope beyond reviewed fixes.

- [ ] **Step 1: Review the complete branch**

  Compare \`main...HEAD\` with
  \`docs/superpowers/specs/2026-07-14-postgres-api-repositories-design.md\`.
  Resolve any Critical, P1, or P2 finding with a focused regression test before
  integration.

- [ ] **Step 2: Re-run the feature-branch quality gate**

  Run:

 ~~~powershell
  $ErrorActionPreference = "Stop"
 & .\.venv\Scripts\python.exe -m ruff check .
  if ($LASTEXITCODE -ne 0) { throw "Ruff lint failed." }
 & .\.venv\Scripts\python.exe -m ruff format --check .
  if ($LASTEXITCODE -ne 0) { throw "Ruff format check failed." }
 & .\.venv\Scripts\python.exe -m pytest -q
  if ($LASTEXITCODE -ne 0) { throw "pytest failed." }
 git status --short --branch
 ~~~

  Expected: zero Ruff issues, all tests passing, and a clean feature branch.

- [ ] **Step 3: Synchronize `main` and rebase the feature branch before integration**

  Run Steps 3 through 5 in one PowerShell session so the bounded native-command
  helper remains available. The helper captures output, gives every network
  command a deadline, and kills only the child process it started:

  ~~~powershell
  $ErrorActionPreference = "Stop"

  function Invoke-BoundedNative {
      param(
          [Parameter(Mandatory)] [string]$FilePath,
          [Parameter(Mandatory)] [string[]]$ArgumentList,
          [Parameter(Mandatory)] [int]$TimeoutSeconds
      )
      $stdout = New-TemporaryFile
      $stderr = New-TemporaryFile
      try {
          $process = Start-Process `
              -FilePath $FilePath `
              -ArgumentList $ArgumentList `
              -PassThru `
              -WindowStyle Hidden `
              -RedirectStandardOutput $stdout `
              -RedirectStandardError $stderr
          if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
              Stop-Process -Id $process.Id -Force
              throw "$FilePath exceeded $TimeoutSeconds seconds."
          }
          $standardOutput = Get-Content -Raw -LiteralPath $stdout
          $standardError = Get-Content -Raw -LiteralPath $stderr
          if ($process.ExitCode -ne 0) {
              throw "$FilePath failed with exit code $($process.ExitCode): $standardError"
          }
          return $standardOutput
      } finally {
          Remove-Item $stdout, $stderr -ErrorAction SilentlyContinue
      }
 }

  function Invoke-CheckedNative {
      param(
          [Parameter(Mandatory)] [string]$FilePath,
          [Parameter(Mandatory)] [string[]]$ArgumentList
      )
      $output = & $FilePath @ArgumentList
      if ($LASTEXITCODE -ne 0) {
          throw "$FilePath failed with exit code $LASTEXITCODE."
      }
      return $output
  }

 function Invoke-BoundedGh {
      param(
          [Parameter(Mandatory)] [string[]]$ArgumentList,
          [Parameter(Mandatory)] [int]$TimeoutSeconds
      )
      try {
          return Invoke-BoundedNative -FilePath gh -ArgumentList $ArgumentList -TimeoutSeconds $TimeoutSeconds
      } catch {
          if ($_.Exception.Message -notmatch "exceeded|Could not resolve|Failed to connect|Connection timed out|Network is unreachable") { throw }
          $hadHttpProxy = Test-Path Env:HTTP_PROXY
          $hadHttpsProxy = Test-Path Env:HTTPS_PROXY
          $previousHttpProxy = $env:HTTP_PROXY
          $previousHttpsProxy = $env:HTTPS_PROXY
          try {
              $env:HTTP_PROXY = "http://127.0.0.1:10808"
              $env:HTTPS_PROXY = "http://127.0.0.1:10808"
              return Invoke-BoundedNative -FilePath gh -ArgumentList $ArgumentList -TimeoutSeconds $TimeoutSeconds
          } finally {
              if ($hadHttpProxy) { $env:HTTP_PROXY = $previousHttpProxy }
              else { Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue }
              if ($hadHttpsProxy) { $env:HTTPS_PROXY = $previousHttpsProxy }
              else { Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue }
          }
      }
  }

  $networkFailurePattern = "exceeded 60 seconds|Could not resolve|Failed to connect|Connection timed out|Network is unreachable"

  $rootWorktree = "D:\Github项目\UrbanFlow-AU"
  $featureWorktree = "D:\Github项目\UrbanFlow-AU\.worktrees\postgres-api-repositories"
  $rootBranch = (Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $rootWorktree, "branch", "--show-current"
  )).Trim()
  if ($rootBranch -ne "main") {
     throw "Root worktree must already be on main before integration."
 }
  $rootChanges = @(Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $rootWorktree, "status", "--porcelain"
  ))
 if ($rootChanges.Count -ne 0) {
     throw "Root worktree is not clean; do not switch or merge."
 }
  $featureBranch = (Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $featureWorktree, "branch", "--show-current"
  )).Trim()
  if ($featureBranch -ne "codex/postgres-api-repositories") {
     throw "Feature worktree is not on codex/postgres-api-repositories."
 }
  $featureChanges = @(Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $featureWorktree, "status", "--porcelain"
  ))
 if ($featureChanges.Count -ne 0) {
     throw "Feature worktree is not clean; commit or resolve changes before rebasing."
 }
 try {
     Invoke-BoundedNative -FilePath git -ArgumentList @(
          "-C", $rootWorktree, "fetch", "origin"
     ) -TimeoutSeconds 60 | Out-Host
 } catch {
      if ($_.Exception.Message -notmatch $networkFailurePattern) { throw }
     Invoke-BoundedNative -FilePath git -ArgumentList @(
         "-c", "http.proxy=http://127.0.0.1:10808",
          "-C", $rootWorktree, "fetch", "origin"
     ) -TimeoutSeconds 60 | Out-Host
 }
  Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $rootWorktree, "merge", "--ff-only", "origin/main"
  ) | Out-Host
  $localMain = (Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $rootWorktree, "rev-parse", "main"
  )).Trim()
  $remoteMain = (Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $rootWorktree, "rev-parse", "origin/main"
  )).Trim()
  if ($localMain -ne $remoteMain) {
      throw "Local main differs from origin/main; do not integrate unrelated commits."
  }
  Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $featureWorktree, "rebase", "main"
  ) | Out-Host
  Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $featureWorktree, "status", "--short", "--branch"
  ) | Out-Host
  ~~~

  Expected: local `main` equals `origin/main`, and the feature branch is clean
  and rebased on that `main`. The proxy fallback is one command only and does
  not write a persistent Git configuration. If the rebase stops on a conflict,
  do not merge or push: resolve the conflicting behavior with a focused
  regression test, run `git add <resolved-files>` and `git rebase --continue`,
  then repeat Task 6 Step 2 before proceeding.

- [ ] **Step 4: Re-run the rebased feature-branch gate**

  From `D:\Github项目\UrbanFlow-AU\.worktrees\postgres-api-repositories`, run:

 ~~~powershell
  $ErrorActionPreference = "Stop"
 & .\.venv\Scripts\python.exe -m ruff check .
  if ($LASTEXITCODE -ne 0) { throw "Ruff lint failed." }
 & .\.venv\Scripts\python.exe -m ruff format --check .
  if ($LASTEXITCODE -ne 0) { throw "Ruff format check failed." }
 & .\.venv\Scripts\python.exe -m pytest -q
  if ($LASTEXITCODE -ne 0) { throw "pytest failed." }
 git status --short --branch
 ~~~

  Expected: zero Ruff issues, all tests passing, and a clean rebased feature
  branch.

- [ ] **Step 5: Fast-forward, push only `main`, verify CI, and clean up**

  Continue in the same PowerShell session from Step 3 and change the current
  shell to `$rootWorktree` before running the merged-main gate:

  ~~~powershell
  Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $rootWorktree, "merge", "--ff-only", "codex/postgres-api-repositories"
  ) | Out-Host
 Set-Location $rootWorktree
  Invoke-CheckedNative -FilePath .\.venv\Scripts\python.exe -ArgumentList @(
      "-m", "ruff", "check", "."
  ) | Out-Host
  Invoke-CheckedNative -FilePath .\.venv\Scripts\python.exe -ArgumentList @(
      "-m", "ruff", "format", "--check", "."
  ) | Out-Host
  Invoke-CheckedNative -FilePath .\.venv\Scripts\python.exe -ArgumentList @(
      "-m", "pytest", "-q"
  ) | Out-Host
  $pushArguments = @("-C", $rootWorktree, "push", "origin", "main")
 try {
      Invoke-BoundedNative -FilePath git -ArgumentList $pushArguments -TimeoutSeconds 60 | Out-Host
 } catch {
      if ($_.Exception.Message -match "non-fast-forward|fetch first") {
          throw "origin/main advanced; do not force-push. Repeat Task 6 Steps 3 and 4."
      }
      if ($_.Exception.Message -notmatch $networkFailurePattern) { throw }
      try {
          Invoke-BoundedNative -FilePath git -ArgumentList @(
              "-c", "http.proxy=http://127.0.0.1:10808",
              "-C", $rootWorktree, "push", "origin", "main"
          ) -TimeoutSeconds 60 | Out-Host
      } catch {
          if ($_.Exception.Message -match "non-fast-forward|fetch first") {
              throw "origin/main advanced; do not force-push. Repeat Task 6 Steps 3 and 4."
          }
          throw
      }
 }
  $head = (Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $rootWorktree, "rev-parse", "HEAD"
  )).Trim()
  $ciDiscoveryDeadline = (Get-Date).AddMinutes(5)
 $run = $null
  while ((Get-Date) -lt $ciDiscoveryDeadline -and $null -eq $run) {
      Write-Output "Waiting for GitHub Actions run for $head..."
      $runsJson = Invoke-BoundedGh -ArgumentList @(
         "run", "list", "--repo", "PoorJeff/UrbanFlow-AU",
         "--branch", "main", "--limit", "20",
         "--json", "databaseId,headSha"
     ) -TimeoutSeconds 60
     $runs = @($runsJson | ConvertFrom-Json)
     $run = @($runs | Where-Object { $_.headSha -eq $head } | Select-Object -First 1)[0]
     if ($null -eq $run) { Start-Sleep -Seconds 5 }
 }
 if ($null -eq $run) {
     throw "No GitHub Actions run for main commit $head appeared within five minutes."
 }
  $ciCompletionDeadline = (Get-Date).AddMinutes(15)
  $ciResult = $null
  while ((Get-Date) -lt $ciCompletionDeadline) {
      $ciJson = Invoke-BoundedGh -ArgumentList @(
          "run", "view", "$($run.databaseId)", "--repo", "PoorJeff/UrbanFlow-AU",
          "--json", "status,conclusion"
      ) -TimeoutSeconds 60
      $ciResult = $ciJson | ConvertFrom-Json
      Write-Output "GitHub Actions $($run.databaseId): status=$($ciResult.status), conclusion=$($ciResult.conclusion)"
      if ($ciResult.status -eq "completed") { break }
      Start-Sleep -Seconds 5
  }
  if ($null -eq $ciResult -or $ciResult.status -ne "completed") {
      throw "GitHub Actions run $($run.databaseId) did not complete within 15 minutes."
  }
  if ($ciResult.conclusion -ne "success") {
      throw "GitHub Actions run $($run.databaseId) concluded $($ciResult.conclusion)."
  }
  Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $rootWorktree, "worktree", "remove", $featureWorktree
  ) | Out-Host
  Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $rootWorktree, "worktree", "prune"
  ) | Out-Host
  Invoke-CheckedNative -FilePath git -ArgumentList @(
      "-C", $rootWorktree, "branch", "-d", "codex/postgres-api-repositories"
  ) | Out-Host
 ~~~

  Expected: the merge is a fast-forward from the current remote `main`; local
  quality checks pass; CI polling prints a status at least every bounded
  request cycle, selects only the run whose `headSha` equals local `main` HEAD,
  and requires `conclusion == "success"` before the feature worktree and local
  feature branch are removed. A non-fast-forward push never uses force and
  returns to synchronization and rebase.

## Plan self-review

### Spec coverage

- Adapter behavior, status \`A\` semantics, SQL queries, error mapping, and
  timezone behavior are covered in Task 1.
- Configuration semantics, no-connection startup, malformed versus unreachable
  URL behavior, injected-service precedence, and ambient-environment isolation
  are covered in Task 2.
- List, lookup, and history HTTP unavailable-data paths are covered in Task 3.
- The required opt-in real PostgreSQL read smoke and temporary-schema safety
  are covered in Task 4.
- README and requirements status alignment are covered in Task 5.
- branch review, merged-main gate, push, CI, and cleanup are covered in Task 6.

### Placeholder scan

This plan contains no unresolved placeholders, unspecified validation, or
implicit test steps. Every production change has a named interface, focused
RED/GREEN command, and verification command.

### Type consistency

\`PostgresSensorHistoryRepository\` implements the existing
\`SensorRepository\` and \`HistoryRepository\` signatures. The default builder
returns \`ApiServices\`; its instance is passed unchanged through
\`create_app(services=...)\`. The smoke uses the same repository constructor
and returns JSON-serializable primitive fields.
