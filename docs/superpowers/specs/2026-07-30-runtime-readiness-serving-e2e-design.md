# Runtime Readiness and Serving E2E Design

Date: 2026-07-30

## Goal

Make the configured UrbanFlow AU serving path truthful and verifiable without
changing public API routes, adding a deployment stack, or claiming model
quality.

This slice makes GET /health report real configuration and lightweight runtime
readiness. It also adds an opt-in smoke command that exercises an isolated
PostgreSQL schema, a real temporary LightGBM artifact, Uvicorn HTTP, and the
Dashboard typed API client.

## Context and problem

The repository already has PostgreSQL-backed sensor/history reads, an
artifact-backed LightGBM provider, a typed FastAPI boundary, and a Streamlit
Dashboard. The default app factory correctly wires a configured repository and
provider, but leaves ApiServices.health at the static default_health
implementation. A configured service therefore still reports every optional
component as unconfigured.

Existing opt-in PostgreSQL and LightGBM smoke commands validate their adapters
independently. Neither starts FastAPI with normal environment configuration nor
calls it through the Dashboard client. The normal test suite and CI must remain
offline and must not require PostgreSQL, model artifacts, network access, or a
running Streamlit process.

The local raw snapshot covers only one day. Generating a real multi-month
supervised training dataset is a separate snapshot-to-evidence slice, not part
of this runtime slice.

## Scope

### In scope

- runtime health state derived from configured dependencies;
- a lightweight read-only PostgreSQL data-cutoff probe performed only during a
  health request;
- explicit operator-controlled data-age semantics;
- model artifact availability and model-version reporting;
- a bounded opt-in isolated serving E2E smoke command;
- Dashboard wording that accurately reflects readiness;
- unit tests, manual smoke coverage, and README instructions.

### Out of scope

- Docker, Docker Compose, deployment, hosted services, or CI PostgreSQL;
- raw-data download changes and the supervised_rows.csv builder;
- real evaluation evidence, MLflow changes, model registry, retraining, or
  committed model artifacts;
- new API routes, schema migrations, prediction persistence, weather,
  Evidently, alerts, or Dashboard pages;
- forecast fallbacks, mock user-facing forecasts, and production-performance
  claims.

## Alternatives considered

### 1. Readiness, training-data construction, and Docker in one change

Rejected. Those changes cross API, modeling, ingestion, packaging, and
Dashboard boundaries. They violate the project small-slice workflow and make
failures hard to diagnose.

### 2. Treat any successful latest-row query as fresh

Rejected. A data cutoff can be known while being stale. Calling it fresh
without an explicit operator threshold would be misleading for historical demo
data.

### 3. Selected: runtime readiness plus bounded serving smoke

Selected. This fixes the most visible trust gap and proves that already-built
service pieces work together. The next slice can add reproducible real-data and
model evidence without mixing it with runtime wiring.

## Architecture

### Health dependency seam

Add a narrow internal protocol in urbanflow.api.services:

~~~
class DataReadinessRepository(Protocol):
    def get_latest_observed_at(self) -> datetime | None: ...
~~~

PostgresSensorHistoryRepository implements this protocol with one read-only
SELECT max(pedestrian_hourly_fact.observed_at) query. It retains its existing
short-lived session ownership and converts SQLAlchemy errors to
DataStoreUnavailableError. Empty tables return None, not a fabricated
timestamp.

No normal request changes repository behavior. The new method is used only by
health.

### RuntimeHealthService

Add a RuntimeHealthService callable in urbanflow.api.services. The app factory
constructs it from:

- an optional DataReadinessRepository;
- the resolved model component status;
- the real artifact manifest model version when loading succeeded;
- an optional maximum data age;
- an injectable UTC clock for deterministic tests.

It must not connect to PostgreSQL when constructed. It performs one readiness
query only when GET /health calls it.

The existing injected ApiServices(health=...) seam remains unchanged, so route
tests and callers can still provide an explicit HealthResult.

### Environment configuration

Add URBANFLOW_API_MAX_DATA_AGE_HOURS as an optional positive integer.

- unset or blank: data age is not configured;
- a positive integer: the latest observation must be no older than that many
  elapsed UTC hours;
- zero, negative, non-integer, or malformed values: fail app construction with
  a clear configuration error.

This setting is evaluated only after a data cutoff is successfully read. It
does not alter data reads or forecasts. Health never exposes a database URL,
artifact path, or underlying exception text.

### Component and overall status semantics

The response model and route remain unchanged.

| Situation | data_store | data_freshness | model_provider | Overall HTTP result |
| --- | --- | --- | --- | --- |
| No database URL | unconfigured | unconfigured | unconfigured | degraded, 200 |
| Database readable, no observations | available | unavailable | based on artifact | degraded, 200 |
| Database readable, cutoff known, age threshold unset | available | unconfigured | based on artifact | degraded, 200 |
| Database readable, cutoff within threshold, valid artifact | available | available | available | ok, 200 |
| Database readable, cutoff stale or later than health clock | available | unavailable | based on artifact | degraded, 200 |
| Database probe raises DataStoreUnavailableError | unavailable | unavailable | based on artifact | unavailable, 503 |
| Artifact absent or ignored because no database is configured | based on database | based on database | unconfigured | degraded, 200 unless database is unavailable |
| Artifact configured but invalid | based on database | based on database | unavailable | degraded, 200 unless database is unavailable |

In this table, based on artifact means available only after a valid artifact
loaded; it otherwise means unconfigured when no artifact path is usable.

data_cutoff_at is the database real latest observation timestamp whenever one
is available. It is never substituted with an artifact training cutoff.

Artifact availability does not prove that every requested sensor has 168
contiguous observations or holiday-calendar coverage. The forecast provider
continues to enforce those request-specific conditions.

### App-factory wiring

create_default_services keeps its existing safe order:

1. No database URL returns empty repositories and a no-I/O degraded health
   service.
2. A valid database URL builds an Engine and repository without opening a
   session.
3. A valid local artifact is loaded only when the database is configured.
4. The function builds RuntimeHealthService from the resulting repository,
   artifact result, and data-age setting.

An invalid artifact remains a non-fatal forecast degradation: sensor and
history reads can still use PostgreSQL. A malformed database URL or malformed
data-age setting remains an application configuration error.

### Opt-in configured-serving smoke

Add a manual command:

~~~
python scripts/smoke_test_serving_e2e.py
~~~

It requires URBANFLOW_SMOKE_DATABASE_URL and never falls back to
URBANFLOW_DATABASE_URL. It follows this bounded lifecycle:

1. Validate or generate a safe temporary PostgreSQL schema name.
2. Create the schema and existing core tables, then seed one active sensor and
   192 contiguous hourly observations ending no more than one hour before the
   smoke clock.
3. Build a small temporary supervised frame and a real local LightGBM artifact
   using the existing artifact exporter. The synthetic values are test
   infrastructure only and are never presented as model-performance evidence.
4. Construct a child-only database URL with the safe schema as PostgreSQL
   search path. Start python -m uvicorn urbanflow.api.app:app on loopback with
   a dynamically selected port and only the child database URL, artifact path,
   and data-age threshold.
5. Poll health with a fixed startup deadline. Then use a real
   DashboardApiClient against that loopback origin to call health, active
   sensors, returned history, and a 24-hour forecast.
6. Assert a 200 ok health result with four available components, the seeded
   cutoff, the artifact model version, one active sensor, ordered history, and
   24 finite non-negative predictions in direct horizon order.
7. In finally, terminate Uvicorn within a fixed shutdown deadline, remove the
   temporary schema, dispose database resources, and remove temporary
   artifacts. On failure, emit bounded redacted diagnostic output only.

The smoke does not launch Streamlit. Calling the real DashboardApiClient
proves the Dashboard HTTP integration boundary, while existing offline
Streamlit tests prove rendering. Full multi-process Dashboard startup belongs
to the later Docker Compose slice.

The Dashboard copy will describe health as a returned readiness signal, not as
a guarantee that every selected sensor can satisfy the 168-hour forecast-input
contract.

## File boundaries

| File | Responsibility |
| --- | --- |
| src/urbanflow/api/services.py | Readiness protocol, runtime health service, status calculation |
| src/urbanflow/api/postgres.py | Latest-observation query and SQLAlchemy error translation |
| src/urbanflow/api/app.py | Environment-to-health wiring and artifact status/version capture |
| src/urbanflow/api/serving_e2e_smoke.py | Isolated schema seed, artifact, Uvicorn lifecycle, client assertions, cleanup |
| scripts/smoke_test_serving_e2e.py | Thin executable entry point |
| tests/unit/api/test_health.py | Health state and response-status tests |
| tests/unit/api/test_app.py | Lazy app-factory wiring and configuration tests |
| tests/unit/api/test_postgres_repositories.py | Latest-observation query and failure mapping |
| tests/unit/api/test_serving_e2e_smoke.py | Offline parser, URL isolation, timeout, and cleanup tests |
| tests/unit/dashboard/test_today.py | Updated truthful readiness wording assertion |
| README.md | Runtime configuration matrix and manual serving-smoke instructions |

No database model, migration, public response schema, Dashboard route, or
feature/modeling file changes in this slice.

## Error handling and safety

- Health probe storage exceptions become component statuses; database outages
  never leak SQLAlchemy details in an HTTP response.
- The smoke validates schema names before DDL and drops only the schema it
  created.
- Child-process environment is constructed explicitly and does not inherit a
  developer accidental serving configuration.
- Uvicorn startup, request, and shutdown have finite deadlines. A failed
  process is terminated before the command returns.
- The manual smoke is not added to CI and never downloads data, modifies the
  public schema, writes model artifacts into the repository, or contacts
  Melbourne Open Data.

## Testing and acceptance criteria

Routine tests remain deterministic and offline.

1. Default app construction has no database session, artifact read, or network
   access and returns the existing degraded health shape.
2. Fake readiness repositories cover fresh, stale, empty, unconfigured, and
   failed data-store cases with a fixed clock.
3. Valid configured repository plus valid artifact reports ok, the real
   manifest model version, and real cutoff only when a valid data-age threshold
   is satisfied.
4. Invalid artifact retains sensor/history service and reports only the model
   component unavailable.
5. Malformed data-age configuration fails explicitly rather than silently
   choosing a threshold.
6. PostgreSQL adapter tests cover latest cutoff, empty result, timezone-aware
   result, and SQLAlchemy failure mapping.
7. E2E smoke unit tests cover missing smoke URL, safe schema validation,
   schema-scoped child configuration, bounded startup failure, nonzero failure
   exit, and cleanup without a live database.
8. A manually configured smoke proves real PostgreSQL, artifact, Uvicorn, and
   Dashboard client integration and prints a non-sensitive JSON result.
9. Ruff check, Ruff format check, pytest, the existing default Uvicorn smoke,
   and a clean Git status pass before integration.

## Follow-up slice

The next slice builds a snapshot-first, manifest-verified supervised_rows.csv
workflow and its provenance/evidence metadata. It will use at least five
complete months for a three-validation-month evaluation and remains separate
from Docker and monitoring work.
