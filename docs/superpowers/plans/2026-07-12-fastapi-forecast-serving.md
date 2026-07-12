# FastAPI Forecast Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` and
> `superpowers:test-driven-development`. Complete each task with a focused
> RED/GREEN cycle, task review, and commit before starting the next task.

**Goal:** Add the first honest, typed FastAPI boundary for UrbanFlow AU without
loading a production model, contacting Melbourne Open Data, or requiring
PostgreSQL, MLflow, or model artifacts in default tests.

**Architecture:** Build the API inside `src/urbanflow/api` with a FastAPI app
factory, Pydantic schemas, domain/service protocols, explicit dependency
injection, and thin routers. Runtime defaults expose health and an empty sensor
catalog while forecast and metrics failures remain explicit until real
providers are configured.

**Tech Stack:** Python 3.11+, FastAPI 0.139.x, Pydantic 2, Uvicorn 0.51.x,
httpx/FastAPI TestClient, pytest, Ruff.

## Global Constraints

- Business routes use `/api/v1`; `/health` remains unversioned.
- Runtime dependencies are `fastapi>=0.139,<1` and
  `uvicorn[standard]>=0.51,<1`.
- Default startup must not train a model, load a model artifact, connect to
  PostgreSQL, contact Melbourne Open Data, or require an MLflow server.
- No fake runtime predictions: an absent model provider returns
  `503 model_unavailable` before sensor lookup.
- Project-raised errors use
  `{"error":{"code":"...","message":"...","details":[]}}`.
- Health aggregate status values are exactly `ok`, `degraded`, and
  `unavailable`; responses include service, package/API version, UTC generation
  time, component records, nullable model version, and nullable data cutoff.
- History ranges require timezone-aware timestamps, use inclusive `start` and
  exclusive `end`, and are capped at 31 days.
- Forecast horizons are integers from 1 through 24, default 24; a provider is
  called once for the requested direct multi-horizon batch; returned rows are
  ordered and predicted counts are clipped to zero.
- `URBANFLOW_API_METRICS_PATH` is the only runtime configuration added in this
  slice. It points to an existing Ridge or LightGBM evaluation summary.
- If an evaluation summary has no artifact version, `model_version` is JSON
  `null`; never fabricate a version, run id, or tracking URI.
- Focused API test output must be warning-free. The existing full suite's 91
  known pandas/NumPy deprecation warnings are pre-existing and explicitly out
  of scope; this slice must not increase that count.
- Do not implement database reads, model serialization/registry, Streamlit,
  Evidently, Docker Compose, Parquet migration, Ridge internals, duplicate
  upsert handling, or bulk-load refactors.

---

### Task 1: API foundation, typed schemas, errors, and health

**Files:**
- Modify: `pyproject.toml`
- Create: `src/urbanflow/api/__init__.py`
- Create: `src/urbanflow/api/app.py`
- Create: `src/urbanflow/api/dependencies.py`
- Create: `src/urbanflow/api/errors.py`
- Create: `src/urbanflow/api/schemas.py`
- Create: `src/urbanflow/api/services.py`
- Create: `src/urbanflow/api/routers/__init__.py`
- Create: `src/urbanflow/api/routers/health.py`
- Test: `tests/unit/api/test_app.py`
- Test: `tests/unit/api/test_health.py`

**Interfaces:**
- Produce `create_app(*, services: ApiServices | None = None) -> FastAPI` and
  module-level `app`.
- Produce `ApiServices`, `HealthService`, `HealthResult`, `ComponentHealth`,
  and `UrbanFlowApiError`.
- Produce Pydantic health and error response schemas used by later routers.

**Behavior:**
- Add the exact dependency bounds from Global Constraints.
- Mount `/health` and establish the app-factory/router inclusion seam. Tasks 2
  through 4 add the four business paths; do not create placeholder endpoints.
- Default health includes `api_process`, `model_provider`, `data_store`, and
  `data_freshness`; unconfigured optional components make the aggregate status
  `degraded` with HTTP 200.
- An injected `ok` result returns HTTP 200 with complete metadata.
- An injected `HealthResult(status="unavailable")` returns HTTP 503.
- Register one exception handler for project errors. Leave FastAPI/Pydantic's
  own request-validation response shape unchanged.

**TDD cycle:**
1. Write app/OpenAPI and health response tests before adding API production
   files; verify collection/import fails for the missing package.
2. Implement the minimum schemas, service container, app factory, health
   router, and exception handler.
3. Re-run focused tests, then Ruff and the full suite before committing.

### Task 2: Sensor catalog and bounded history

**Files:**
- Modify: `src/urbanflow/api/dependencies.py`
- Modify: `src/urbanflow/api/schemas.py`
- Modify: `src/urbanflow/api/services.py`
- Create: `src/urbanflow/api/routers/sensors.py`
- Test: `tests/unit/api/test_sensors.py`
- Test: `tests/unit/api/test_history.py`

**Interfaces:**
- Produce `SensorRecord`, `HistoryRecord`, `SensorRepository`,
  `HistoryRepository`, and `HistoryService`.
- `SensorRepository.list_sensors(active_only: bool)` returns sensor records;
  `get_sensor(location_id: int)` returns one record or `None`.
- `HistoryRepository.get_history(location_id, start, end)` returns hourly
  records without performing HTTP-layer validation.

**Behavior:**
- `GET /api/v1/sensors?active_only=true` returns `{data, meta}`; default empty
  repository returns 200 with an empty list.
- `GET /api/v1/sensors/{location_id}/history` validates location ids, timezone
  awareness, ordering, and the 31-day cap; unknown sensors return
  `404 sensor_not_found`.
- Repository failures explicitly raised as data-store failures become
  `503 data_store_unavailable`; history rows are sorted by `observed_at`.
- Tests use complete in-memory fakes and assert HTTP behavior, not fake calls.

**TDD cycle:**
1. Add focused sensor/history success and failure tests and verify expected
   route/attribute failures.
2. Add the minimum records, protocols, services, router, and dependency
   wiring.
3. Re-run focused tests, Ruff, and the full suite before committing.

### Task 3: Direct multi-horizon forecast service

**Files:**
- Modify: `src/urbanflow/api/dependencies.py`
- Modify: `src/urbanflow/api/schemas.py`
- Modify: `src/urbanflow/api/services.py`
- Create: `src/urbanflow/api/routers/forecasts.py`
- Test: `tests/unit/api/test_forecasts.py`

**Interfaces:**
- Produce `ForecastModelProvider`, `ForecastBatch`, `ForecastPrediction`, and
  concrete `ForecastService`.
- `ForecastModelProvider.predict(location_id: int, horizon: int)` is called
  once and returns provider/model metadata plus all requested prediction rows.

**Behavior:**
- `GET /api/v1/sensors/{location_id}/forecast?horizon=24` accepts 1..24.
- Provider availability is checked before sensor lookup; absent provider
  returns `503 model_unavailable` for every location id.
- With a provider, unknown sensors return `404 sensor_not_found`.
- Validate that provider rows cover horizons `1..horizon` exactly once; sort by
  horizon and clip each `predicted_count` to `max(value, 0.0)`.
- Preserve timezone-aware provider timestamps and include model name/version,
  generation time, forecast origin, and data cutoff in the response.

**TDD cycle:**
1. Add tests for default/boundary/invalid horizons, missing provider, unknown
   sensor, one provider call, output ordering, metadata, and clipping.
2. Implement the minimal provider protocol, service, router, and wiring.
3. Re-run focused tests, Ruff, and the full suite before committing.

### Task 4: Evaluation-summary metrics endpoint

**Files:**
- Modify: `src/urbanflow/api/dependencies.py`
- Modify: `src/urbanflow/api/schemas.py`
- Modify: `src/urbanflow/api/services.py`
- Create: `src/urbanflow/api/routers/models.py`
- Test: `tests/unit/api/test_model_metrics.py`

**Interfaces:**
- Produce `ModelMetadataProvider` and `EvaluationSummaryMetadataProvider`.
- `EvaluationSummaryMetadataProvider` consumes an optional `Path` and returns
  final-test metrics in the API response contract.

**Behavior:**
- `GET /api/v1/model/metrics` returns MAE, RMSE, WAPE, Seasonal Naive WAPE,
  relative WAPE improvement, final-test window, and evaluation source.
- Infer `model_name` only when exactly one supported comparison key exists:
  `ridge_wape` or `lightgbm_wape`; ambiguous or unsupported summaries fail.
- Missing path, missing file, unreadable/invalid JSON, or missing required
  summary fields return `503 metrics_unavailable`.
- Return `model_version`, `mlflow_run_id`, `mlflow_tracking_uri`, and
  `report_path` as `null` when absent from the source summary.
- Read the configured file lazily so a bad path never prevents `/health` from
  responding.

**TDD cycle:**
1. Add Ridge/LightGBM and all unavailable/invalid summary tests; verify the
   missing provider/route behavior fails first.
2. Implement the provider, response schemas, route, and environment wiring.
3. Re-run focused tests, Ruff, and the full suite before committing.

### Task 5: User documentation and contract alignment

**Files:**
- Modify: `README.md`
- Modify: `urbanflow-au_requirements.md`

**Behavior:**
- Document `python -m uvicorn urbanflow.api.app:app --reload`, `/health`,
  OpenAPI docs, and example business endpoint requests.
- State that runtime forecast requests return `503 model_unavailable` until an
  explicit provider is wired and that no production-quality claim is made.
- Document optional `URBANFLOW_API_METRICS_PATH` and its nullable version/run
  metadata behavior.
- Align requirements sections 10 and 13 with `/api/v1` nested business routes
  and `src/urbanflow/api`; do not rewrite unrelated requirements.
- Update the roadmap to say only the first FastAPI contract boundary is in
  place; Streamlit and Evidently remain future work.

**Verification:**
1. Run focused API tests, Ruff lint, Ruff format check, and the full pytest
   suite; API changes must add no warnings beyond the 91 known warnings.
2. Start Uvicorn locally and verify `/health` and `/openapi.json` over HTTP.
3. Verify a clean branch status and commit the documentation.

### Task 6: Whole-branch review and integration

**Behavior:**
- Review the complete branch against the approved design, this plan, and the
  explicit deferrals.
- Resolve all Critical and Important findings with focused tests and re-review.
- Run a fresh final quality gate and HTTP smoke.
- Fast-forward merge `codex/fastapi-forecast-serving` into `main`, run the same
  gate again on `main`, push only `main`, verify the new GitHub Actions run,
  then remove the worktree and local feature branch.
