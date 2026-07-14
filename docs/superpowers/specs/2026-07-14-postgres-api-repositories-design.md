# PostgreSQL API Repositories Design

Date: 2026-07-14

## Goal

Make the existing sensor catalog and history HTTP contracts read real,
previously loaded PostgreSQL data without changing forecast serving, model
artifacts, Dashboard work, or database schema.

With a configured database URL, these endpoints will use the persisted
UrbanFlow tables:

- \`GET /api/v1/sensors\`
- \`GET /api/v1/sensors/{location_id}/history\`

Without a configured database URL, the API must retain the current safe,
network-free default behavior.

## Current project context

UrbanFlow already has:

- a FastAPI app factory, typed endpoint schemas, repository protocols, and
  project-level error responses;
- \`SensorRepository\` and \`HistoryRepository\` protocols in
  \`src/urbanflow/api/services.py\`;
- PostgreSQL persistence models for \`sensor_dim\` and
  \`pedestrian_hourly_fact\`;
- \`URBANFLOW_DATABASE_URL\`, \`create_database_engine\`, and
  \`create_session_factory\`;
- migration, loader, and optional PostgreSQL persistence smoke coverage.

The API currently injects \`EmptySensorRepository\` and
\`EmptyHistoryRepository\` by default. Consequently, the sensor list is empty
and history requests return \`404 sensor_not_found\`. Forecasts intentionally
remain unavailable until a separate model-artifact and inference-provider
slice is complete.

The source sensor contract uses status codes \`A\` and \`I\`. This slice defines
\`active_only=true\` as exactly \`status == "A"\`; it does not infer activity
from free-form text.

## Selected approach

Add one API-facing SQLAlchemy adapter,
\`PostgresSensorHistoryRepository\`, which implements both existing
repository protocols. It will receive a SQLAlchemy \`sessionmaker\`, open a
short-lived read session for each method call, map ORM rows to the existing API
records, and translate SQLAlchemy read failures into
\`DataStoreUnavailableError\`.

Use an explicit default-services builder in the app layer:

1. no \`URBANFLOW_DATABASE_URL\` or a whitespace-only value: retain empty
   repositories;
2. a non-empty URL: construct an Engine and session factory without issuing a
   database query, then inject one PostgreSQL adapter for both sensor and
   history reads;
3. an invalid non-empty URL: fail application construction with a configuration
   error rather than silently presenting empty data.

Creating a SQLAlchemy Engine does not establish a database connection. The
first connection happens only when an endpoint creates a session and executes
its read query.

## Architecture and file boundaries

~~~text
src/urbanflow/api/
├── app.py                 # build default services from optional DB configuration
├── services.py            # existing protocols and errors stay authoritative
└── postgres.py            # PostgreSQL-to-API read adapter

tests/unit/api/
├── test_postgres_repositories.py
└── test_app.py            # default configuration wiring coverage

scripts/
└── smoke_test_postgres_api.py
~~~

### PostgreSQL adapter

\`src/urbanflow/api/postgres.py\` owns only the translation between SQLAlchemy
database models and API service records. It depends on:

- \`SensorDim\` and \`PedestrianHourlyFact\`;
- \`Session\` / \`sessionmaker\`;
- \`SensorRecord\`, \`HistoryRecord\`, and \`DataStoreUnavailableError\`.

It must not import FastAPI, router code, model-training code, MLflow, or
Melbourne Open Data clients.

The adapter exposes:

~~~python
class PostgresSensorHistoryRepository:
    def list_sensors(self, active_only: bool) -> list[SensorRecord]: ...
    def get_sensor(self, location_id: int) -> SensorRecord | None: ...
    def get_history(
        self,
        location_id: int,
        start: datetime,
        end: datetime,
    ) -> list[HistoryRecord]: ...
~~~

Each method opens and closes its own session. No request writes, commits, or
contacts an external API.

### Query behavior

\`list_sensors(active_only=True)\` selects rows from \`sensor_dim\` where
\`status == "A"\`, ordered by \`location_id\`. With \`active_only=False\`, it
returns every sensor, still ordered by \`location_id\`.

\`get_sensor(location_id)\` selects the sensor by primary key and returns
\`None\` when no row exists.

\`get_history(location_id, start, end)\` selects from
\`pedestrian_hourly_fact\` with:

~~~sql
WHERE location_id = :location_id
  AND observed_at >= :start
  AND observed_at < :end
ORDER BY observed_at ASC
~~~

The adapter passes timezone-aware values through unchanged. The existing
\`HistoryService\` remains the authority for input validation, unknown-sensor
handling, defensive range filtering, and final ordering.

### Application wiring

\`create_app(services=...)\` remains the explicit injection seam used by tests
and callers. When no \`services\` argument is supplied, the app uses a
default-services builder:

- no database URL: \`ApiServices()\`, preserving the current empty
  repositories;
- configured database URL: one
  \`PostgresSensorHistoryRepository(session_factory)\` assigned to both
  \`sensor_repository\` and \`history_repository\`;
- explicit injected services: never overridden by environment configuration.

This slice does not load a forecast provider. The default forecast endpoint
therefore continues to return \`503 model_unavailable\`.

## Error behavior

SQLAlchemy connection, statement, and row-read errors from the adapter are
caught as \`SQLAlchemyError\` and re-raised as
\`DataStoreUnavailableError\`. Existing router and service code maps that
error to the established response:

~~~json
{
  "error": {
    "code": "data_store_unavailable",
    "message": "Sensor data is currently unavailable.",
    "details": []
  }
}
~~~

The endpoint status is \`503\`. A missing sensor remains \`404 sensor_not_found\`;
an empty interval for a known sensor remains a successful \`200\` response with
an empty data list.

This slice deliberately does not add a live database probe to \`/health\`.
The current health response is configuration-oriented, and reporting a
database as available without probing it would be misleading. Database
health and data-freshness checks are deferred to a dedicated health slice.

## Testing strategy

Default tests must not require PostgreSQL, network access, model files, or an
MLflow server.

Repository tests cover:

1. all sensors and deterministic \`location_id\` ordering;
2. \`active_only=true\` filtering for \`A\` versus \`I\`;
3. known and unknown sensors;
4. history start-inclusive/end-exclusive boundaries and ascending timestamps;
5. preservation of aware datetimes while mapping rows;
6. SQLAlchemy failures becoming \`DataStoreUnavailableError\`.

These tests use controlled SQLAlchemy sessions or session fakes and do not
start a PostgreSQL service.

App-factory and endpoint tests cover:

1. no database URL retains the existing empty default repositories;
2. a configured URL creates PostgreSQL-backed default repositories without
   executing a query during app creation;
3. injected \`ApiServices\` still wins over environment configuration;
4. a repository read failure returns the existing \`503
   data_store_unavailable\` response.

Add an opt-in manual PostgreSQL smoke command. It receives the existing
\`URBANFLOW_SMOKE_DATABASE_URL\`, creates an isolated temporary schema, writes
one sensor and one hourly observation through the existing persistence helpers,
then reads them through \`PostgresSensorHistoryRepository\`. It verifies sensor
visibility, \`A\` filtering, and history mapping before dropping the temporary
schema. Routine pytest remains database-free.

## Documentation

Update the README to show:

1. setting \`URBANFLOW_DATABASE_URL\`;
2. loading validated sensor and hourly snapshots into PostgreSQL;
3. starting Uvicorn;
4. calling the live sensor and history endpoints.

The README must still state that real forecast serving needs a separately
configured model artifact and provider.

## Non-goals

This slice does not:

- create or change Alembic migrations;
- add PostgreSQL training reads;
- serialize, register, load, or select model artifacts;
- implement a \`ForecastModelProvider\`;
- change \`/health\` data-store or freshness semantics;
- query Melbourne Open Data at request time;
- add Streamlit, Evidently, Docker Compose, or a Dashboard;
- change existing upsert behavior, batching, Parquet use, or baseline models.

## Alternatives considered

### Add model artifact serving first

Rejected. Direct multi-horizon inference needs trustworthy recent history,
which the current API cannot yet read. Building artifact loading first would
require a second arbitrary history source or a fake runtime input.

### Add a Dashboard before real reads

Rejected. It would primarily display an empty sensor list and unavailable
forecasts, contradicting the requirement that Dashboard views use real pipeline
output.

### Query PostgreSQL directly in route handlers

Rejected. The existing repository protocols already isolate route handlers from
storage details and make unavailable-data paths inexpensive to test.

### Use a status-name heuristic for active sensors

Rejected. Persisted source data uses \`A\` / \`I\`; treating strings such as
\`"Active"\` or \`"active"\` as equivalent would invent a data contract.

## Acceptance criteria

The PostgreSQL API repository slice is complete when:

- an optional database URL wires real sensor and history repositories into the
  default API app;
- no database URL preserves default empty behavior and does not connect to
  PostgreSQL;
- \`active_only=true\` returns only \`status == "A"\`;
- history reads are time-zone-aware, ascending, and use \`[start, end)\`;
- database failures return \`503 data_store_unavailable\`;
- no model, forecast, health-probe, migration, Dashboard, or external-data
  behavior changes;
- unit tests remain network- and PostgreSQL-free;
- the manual PostgreSQL read smoke is opt-in;
- Ruff, format checks, pytest, Uvicorn smoke, and GitHub Actions pass.
