# LightGBM Artifact Forecast Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Export one validated local LightGBM artifact and serve its truthful direct 1–24 hour forecasts through the existing opt-in PostgreSQL FastAPI boundary.

**Architecture:** Keep model serialization, artifact validation, and final local fitting in the modeling layer. Add one API provider that turns exactly 168 recent PostgreSQL observations into Melbourne-local direct-horizon features, then inject it only when both a database URL and a valid local artifact directory are configured. The route, response schema, health semantics, and safe defaults remain unchanged.

**Tech Stack:** Python 3.11+, pandas, scikit-learn, LightGBM, joblib, SQLAlchemy 2, psycopg 3, FastAPI, Pydantic 2, pytest, httpx ASGI transport, Uvicorn, Ruff, GitHub Actions.

## Global Constraints

- Use only the local directory named by URBANFLOW_API_MODEL_ARTIFACT_PATH; validate the raw string before converting it to Path, reject a value containing ://, also reject a Path-normalized URI-like form such as s3:\bucket, and never download or register a model remotely. Do not reject ordinary Windows drive paths such as C:\models\artifact.
- Declare joblib as a direct dependency. The artifact contains a full FittedLightGBMModel, including its fitted scikit-learn preprocessing pipeline; never serialize only LGBMRegressor.
- Artifact schema version is exactly 1. The directory contains exactly manifest.json and model.joblib. Existing destinations are refused, and creation uses a temporary sibling directory followed by a rename.
- A valid artifact uses model_name lightgbm; model_version equals lightgbm- plus the first twelve lowercase hexadecimal characters of model_sha256; all model and source hashes are 64 lowercase hexadecimal characters.
- The artifact supports only DEFAULT_RIDGE_FEATURE_SPEC and feature_timezone Australia/Melbourne. Validate that manifest feature_columns, FittedLightGBMModel.feature_columns, and config.feature_spec.feature_columns are exactly equal and ordered.
- The model configuration fields n_estimators, num_leaves, and min_child_samples are positive integers; learning_rate is finite and positive; max_depth is an integer at least -1; random_state is an integer.
- The exporter fits every eligible non-missing target row in its explicit supervised CSV. It is not a rolling-origin evaluator and does not claim new evaluation metrics.
- A holiday calendar is supplied as a local JSON object with exactly coverage_start, coverage_end, and public_holidays. Dates are ISO dates, coverage is inclusive, holidays are sorted/unique/in-range, and every requested target date must be covered.
- There is no weather source in this slice. Refuse an artifact training frame when any eligible row contains observed temperature, rainfall, or wind_speed data, or when its corresponding missing marker is not true. Runtime uses the existing all-missing weather feature contract.
- Recent history is exactly 168 ascending records, timezone-aware, on UTC-hour boundaries, separated by one UTC hour, and has finite non-negative integral counts. Reject bool, float, negative, duplicate, gapped, naive, and non-hour values.
- Convert validated history instants to the named Australia/Melbourne timezone before feature generation. This preserves local calendar features across PostgreSQL UTC round trips and daylight-saving transitions.
- A provider horizon is a non-boolean Integral in 1 through 24, validated before any repository or model call. Advance forecast target instants in UTC and then convert them to Australia/Melbourne; never use local wall-clock datetime addition across a daylight-saving transition.
- Serving-input failures map to forecast_unavailable, while malformed provider/model output (wrong count or values that cannot become floats) maps to model_unavailable. Do not silently truncate extra model outputs or fabricate a forecast row for them.
- Keep direct multi-horizon semantics: generate all requested horizons from one observed cutoff. Never write a prediction into history or use recursive predictions as lag values.
- Do not change health behavior, routes, Pydantic response models, migrations, ingestion, weather fetching, MLflow registry behavior, Dashboard, Evidently, or Docker Compose.
- With no database URL or a blank URL, do not create an Engine, open a session, read an artifact, or contact any remote service. With a database but no valid artifact, sensor/history reads remain usable and forecast remains the existing 503 model_unavailable.
- Routine tests are offline and require no PostgreSQL server, MLflow server, network, or committed model artifact. Existing pandas/NumPy warnings may remain but must not grow.
- Follow docs/development_workflow.md: work only on codex/lightgbm-artifact-forecast-serving, use TDD, commit each reviewed task conventionally, push only main after successful integration, and use bounded native/network commands at integration time.

## Repository Map

| Path | Responsibility after this slice |
| --- | --- |
| src/urbanflow/modeling/supervised_csv.py | Shared local supervised-CSV parsing and byte-hash helper used by evaluation and artifact export. |
| src/urbanflow/modeling/lightgbm_artifact.py | Versioned manifest, holiday calendar, local artifact export/load, integrity and feature-contract validation. |
| src/urbanflow/modeling/lightgbm_artifact_cli.py | Thin final-fit/export command-line adapter. |
| scripts/export_lightgbm_artifact.py | Executable wrapper for the export CLI. |
| src/urbanflow/api/services.py | Recent-history protocol plus service-level forecast error mapping. |
| src/urbanflow/api/errors.py | Standard 503 forecast_unavailable response constructor. |
| src/urbanflow/api/postgres.py | PostgreSQL implementation of the latest-N history read. |
| src/urbanflow/api/lightgbm_provider.py | Artifact-backed direct forecast provider and strict serving-input checks. |
| src/urbanflow/api/app.py | Explicit environment wiring for one loaded artifact provider. |
| src/urbanflow/api/lightgbm_forecast_smoke.py | Opt-in real PostgreSQL plus local-artifact provider smoke. |
| tests/unit/modeling/ | Artifact format, export CLI, and evaluator-regression tests. |
| tests/unit/api/ | Provider, service/error, repository, app-factory, and smoke unit tests. |
| README.md and urbanflow-au_requirements.md | Honest operator instructions and project-status boundary. |
| .github/workflows/ci.yml | Default-configured bounded Uvicorn health smoke after pytest. |

## Preflight

Run this once from the feature worktree before Task 1. Do not begin code changes if the editable install or quality gate fails. The virtual environment is intentionally local to this worktree and remains ignored by Git.

~~~powershell
$ErrorActionPreference = "Stop"
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    python -m venv .venv
}
& .\.venv\Scripts\python.exe -c "import sys; assert sys.version_info >= (3, 11), sys.version"
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -c "import urbanflow, fastapi, joblib; print(urbanflow.__version__)"
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest -q
git status --short --branch
~~~

Expected: editable import succeeds, current quality checks pass, and the only branch is the clean feature worktree. If package installation exceeds a bounded command deadline, stop that child process, report the install error, and do not substitute PYTHONPATH or the root worktree environment.

---

### Task 1: Add the validated local artifact domain and shared CSV reader

**Files:**

- Create: src/urbanflow/modeling/supervised_csv.py
- Modify: src/urbanflow/modeling/lightgbm_cli.py
- Create: src/urbanflow/modeling/lightgbm_artifact.py
- Modify: pyproject.toml
- Create: tests/unit/modeling/test_supervised_csv.py
- Create: tests/unit/modeling/test_lightgbm_artifact.py
- Modify: tests/unit/modeling/test_lightgbm_cli.py

**Interfaces:**

- Consumes: existing LightGBMModelConfig, FittedLightGBMModel, fit_lightgbm_model, DEFAULT_RIDGE_FEATURE_SPEC, select_training_rows, pandas, and joblib.
- Produces:

~~~python
class SupervisedCsvError(ValueError): ...

def read_supervised_csv(path: Path) -> pd.DataFrame: ...
def sha256_file(path: Path) -> str: ...

class LightGBMArtifactError(ValueError): ...
class LightGBMArtifactSerializationError(RuntimeError): ...

@dataclass(frozen=True, slots=True)
class HolidayCalendar:
    coverage_start: date
    coverage_end: date
    public_holidays: tuple[date, ...]

    @classmethod
    def from_json_file(cls, path: Path) -> HolidayCalendar: ...
    def contains(self, value: date) -> bool: ...
    def to_manifest_fields(self) -> dict[str, object]: ...

@dataclass(frozen=True, slots=True)
class LightGBMArtifactManifest:
    schema_version: int
    model_name: str
    model_version: str
    model_sha256: str
    training_data_sha256: str
    created_at: datetime
    trained_through_at: datetime
    training_row_count: int
    feature_timezone: str
    feature_columns: tuple[str, ...]
    model_config: dict[str, int | float]
    holiday_calendar: HolidayCalendar
    evaluation_summary_path: str | None

@dataclass(frozen=True, slots=True)
class LoadedLightGBMArtifact:
    manifest: LightGBMArtifactManifest
    model: FittedLightGBMModel

def export_lightgbm_artifact(
    supervised_frame: pd.DataFrame,
    *,
    source_csv_sha256: str,
    output_directory: str | Path,
    holiday_calendar: HolidayCalendar,
    model_config: LightGBMModelConfig,
    evaluation_summary_path: str | None = None,
) -> LightGBMArtifactManifest: ...

def load_lightgbm_artifact(path: str | Path) -> LoadedLightGBMArtifact: ...
~~~

- The existing evaluation CLI must use read_supervised_csv and convert SupervisedCsvError to its existing LightGBMEvaluationCliError message so its public exit behavior does not change.

- [ ] **Step 1: Write focused failing reader and artifact tests**

Create tests with a local 192-row supervised CSV helper. The helper must include all existing model feature columns, timezone-aware Australia/Melbourne forecast_origin_at and target_observed_at columns, a non-missing target, all three weather value columns as pd.NA, and all three weather missing-marker columns as True. Write a local holiday JSON helper:

~~~python
def write_holiday_calendar(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "coverage_start": "2025-01-01",
                "coverage_end": "2026-12-31",
                "public_holidays": ["2025-01-27", "2026-01-26"],
            }
        ),
        encoding="utf-8",
    )
    return path

def test_artifact_round_trip_preserves_pipeline_and_manifest(tmp_path: Path) -> None:
    csv_path = write_supervised_csv(tmp_path / "rows.csv")
    frame = read_supervised_csv(csv_path)
    calendar = HolidayCalendar.from_json_file(write_holiday_calendar(tmp_path / "holidays.json"))

    manifest = export_lightgbm_artifact(
        frame,
        source_csv_sha256=sha256_file(csv_path),
        output_directory=tmp_path / "artifact",
        holiday_calendar=calendar,
        model_config=LightGBMModelConfig(n_estimators=5, min_child_samples=1),
    )
    loaded = load_lightgbm_artifact(tmp_path / "artifact")

    assert manifest.model_name == "lightgbm"
    assert manifest.model_version == f"lightgbm-{manifest.model_sha256[:12]}"
    assert manifest.feature_columns == DEFAULT_RIDGE_FEATURE_SPEC.feature_columns
    assert loaded.manifest == manifest
    assert isinstance(loaded.model, FittedLightGBMModel)
    assert (tmp_path / "artifact").iterdir()
    assert {path.name for path in (tmp_path / "artifact").iterdir()} == {
        "manifest.json",
        "model.joblib",
    }
~~~

Add independent tests that assert:

1. read_supervised_csv parses both timestamp columns, preserves the true UTC instants of a Melbourne CSV spanning the +10/+11 daylight-saving offset change, and rejects a missing path, malformed CSV, an unparseable present timestamp column, or a timezone-naive timestamp;
2. sha256_file equals hashlib.sha256(path.read_bytes()).hexdigest();
3. export creates a missing local parent directory, rejects an existing destination, and rejects both a raw path containing :// and a URI-like path already normalized by Path (for example Path("s3://bucket/artifact")) before any artifact write;
4. loader rejects a missing bundle member, an extra bundle member, wrong schema version/model name, uppercase or checksum-mismatched hash, a version not derived from the checksum, a naive created_at or trained_through_at, invalid model-config types/ranges, and feature columns that differ from either fitted-model feature source;
5. HolidayCalendar rejects a wrong key set, invalid ISO date, reversed coverage, duplicate/unsorted/out-of-range holiday, and a coverage miss;
6. export rejects an eligible training row with a real weather value or a false weather marker;
7. the existing LightGBM evaluation CLI still prints its normal summary for a valid CSV after the reader extraction.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/modeling/test_supervised_csv.py tests/unit/modeling/test_lightgbm_artifact.py tests/unit/modeling/test_lightgbm_cli.py -q
~~~

Expected: collection fails because supervised_csv and lightgbm_artifact do not yet exist.

- [ ] **Step 3: Implement the shared reader and integrity helpers**

Create src/urbanflow/modeling/supervised_csv.py. Keep exactly the existing two timestamp names and parse only those columns when present. A source timestamp must be offset-bearing: first validate each original value is parseable and timezone-aware, then normalize the complete column with pd.to_datetime(..., format="mixed", utc=True). This makes a CSV spanning Melbourne's +10/+11 daylight-saving change a timezone-aware UTC series with the same real instants; do not call utc=True on a naive value and silently reinterpret it as UTC. The following is only the structural outline:

~~~python
TIMESTAMP_COLUMNS = ("forecast_origin_at", "target_observed_at")

class SupervisedCsvError(ValueError):
    pass

def read_supervised_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SupervisedCsvError(f"CSV file does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise SupervisedCsvError(f"could not read supervised CSV: {path}") from exc
    for column in TIMESTAMP_COLUMNS:
        if column in frame.columns:
            try:
                frame[column] = parse_offset_aware_timestamp_column(frame[column])
            except (TypeError, ValueError) as exc:
                raise SupervisedCsvError(
                    f"could not parse timestamp column: {column}"
                ) from exc
    return frame

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
~~~

Replace the private reader in lightgbm_cli.py with this helper. At the call boundary, catch SupervisedCsvError and re-raise LightGBMEvaluationCliError with the same text so the existing CLI remains behaviorally stable.

In pyproject.toml, add the direct runtime dependency:

~~~toml
"joblib>=1.2,<2",
~~~

Place it beside LightGBM and scikit-learn rather than relying on scikit-learn's transitive dependency.

- [ ] **Step 4: Implement the artifact format and validator**

Create src/urbanflow/modeling/lightgbm_artifact.py with these fixed constants:

~~~python
ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_MODEL_NAME = "lightgbm"
ARTIFACT_MODEL_FILE_NAME = "model.joblib"
ARTIFACT_MANIFEST_FILE_NAME = "manifest.json"
FEATURE_TIMEZONE = "Australia/Melbourne"
_EXPECTED_BUNDLE_FILES = {
    ARTIFACT_MANIFEST_FILE_NAME,
    ARTIFACT_MODEL_FILE_NAME,
}
~~~

Implement HolidayCalendar.from_json_file by loading a UTF-8 JSON object, requiring exactly the three documented keys, parsing date.fromisoformat values, and converting holidays to a tuple only when it is already sorted, duplicate-free, and inside inclusive coverage. Implement contains as:

~~~python
def contains(self, value: date) -> bool:
    return self.coverage_start <= value <= self.coverage_end
~~~

Implement export_lightgbm_artifact as the following sequence:

1. pass the raw str | Path output argument through one canonical local-path validator before any Path coercion or filesystem write; it rejects :// and URI-like normalized forms such as s3:\bucket while accepting Windows drive paths, then reject an existing output directory and create its missing local parent directory;
2. call select_training_rows, then validate all eligible weather value cells are missing and their three markers are exactly true;
3. require timezone-aware forecast_origin_at values, derive trained_through_at as their maximum, and fit the complete pipeline with fit_lightgbm_model;
4. require fitted_model.config.feature_spec and fitted_model.feature_columns to equal DEFAULT_RIDGE_FEATURE_SPEC and its ordered feature_columns;
5. create a temporary sibling directory with tempfile.mkdtemp(dir=output_directory.parent), joblib.dump the complete fitted model to model.joblib, calculate its SHA-256, and create a manifest that includes the exact CSV hash, UTC created_at, copied holiday calendar, core scalar config, and model version derived from the model bytes;
6. write manifest.json with sorted JSON keys and a trailing newline, validate the temporary bundle through load_lightgbm_artifact, then rename the temporary directory to the absent output path;
7. on serialization or atomic-output failures, remove only that temporary directory and raise LightGBMArtifactSerializationError. On validation/input failures, raise LightGBMArtifactError. Never overwrite an existing output.

The loader must first pass its raw str | Path input through that same canonical local-path validator, then require an existing local directory whose child names equal _EXPECTED_BUNDLE_FILES. It reads manifest JSON, validates every field before deserializing, compares model.joblib bytes to model_sha256, loads only then with joblib.load, requires FittedLightGBMModel, and validates its ordered features/config against the manifest and DEFAULT_RIDGE_FEATURE_SPEC. Return LoadedLightGBMArtifact only after every check passes.

- [ ] **Step 5: Run focused verification**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/modeling/test_supervised_csv.py tests/unit/modeling/test_lightgbm_artifact.py tests/unit/modeling/test_lightgbm_cli.py -q
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/modeling/supervised_csv.py src/urbanflow/modeling/lightgbm_artifact.py src/urbanflow/modeling/lightgbm_cli.py tests/unit/modeling/test_supervised_csv.py tests/unit/modeling/test_lightgbm_artifact.py tests/unit/modeling/test_lightgbm_cli.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/modeling/supervised_csv.py src/urbanflow/modeling/lightgbm_artifact.py src/urbanflow/modeling/lightgbm_cli.py tests/unit/modeling/test_supervised_csv.py tests/unit/modeling/test_lightgbm_artifact.py tests/unit/modeling/test_lightgbm_cli.py
~~~

Expected: focused tests pass, the evaluator CLI regression passes, and Ruff reports no changes.

- [ ] **Step 6: Commit the artifact-domain task**

~~~powershell
git add pyproject.toml src/urbanflow/modeling/supervised_csv.py src/urbanflow/modeling/lightgbm_cli.py src/urbanflow/modeling/lightgbm_artifact.py tests/unit/modeling/test_supervised_csv.py tests/unit/modeling/test_lightgbm_artifact.py tests/unit/modeling/test_lightgbm_cli.py
git commit -m "feat(modeling): add lightgbm artifact format"
~~~

### Task 2: Add the final-fit artifact export CLI

**Files:**

- Create: src/urbanflow/modeling/lightgbm_artifact_cli.py
- Create: scripts/export_lightgbm_artifact.py
- Create: tests/unit/modeling/test_lightgbm_artifact_cli.py

**Interfaces:**

- Consumes: read_supervised_csv, sha256_file, HolidayCalendar, export_lightgbm_artifact, LightGBMModelConfig, and the artifact exceptions from Task 1.
- Produces:

~~~python
class LightGBMArtifactCliError(ValueError): ...

def build_parser() -> argparse.ArgumentParser: ...
def run_artifact_export(
    supervised_csv: Path,
    output_directory: str | Path,
    *,
    holiday_calendar_path: Path,
    model_config: LightGBMModelConfig,
    evaluation_summary_path: str | None,
) -> LightGBMArtifactManifest: ...
def main(argv: Sequence[str] | None = None) -> int: ...
~~~

- [ ] **Step 1: Write failing CLI tests**

Use tmp_path-only inputs. Test a successful invocation with a five-tree model and assert exit code 0, JSON stdout includes model_name, model_version, training_row_count, trained_through_at, and output_directory, and both artifact files exist. Add exact exit-code tests:

~~~python
def test_cli_returns_two_for_invalid_operator_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            str(tmp_path / "missing.csv"),
            str(tmp_path / "artifact"),
            "--holiday-calendar",
            str(tmp_path / "missing-holidays.json"),
        ]
    )

    assert exit_code == 2
    assert "error:" in capsys.readouterr().err
~~~

Also cover a zero n-estimators, non-positive learning rate, malformed calendar, existing destination, an opaque URI-like output such as s3:bucket/artifact, and a monkeypatched joblib.dump that raises OSError. The invalid output path must return 2 and create no artifact. The dump failure must return 1 rather than 2. Assert no test uses a network, a database URL, MLflow, or a committed models directory.

- [ ] **Step 2: Run the focused CLI test and confirm RED**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/modeling/test_lightgbm_artifact_cli.py -q
~~~

Expected: collection fails because lightgbm_artifact_cli and the script wrapper do not exist.

- [ ] **Step 3: Implement the thin CLI**

Create the parser with positional supervised_csv and output_directory arguments plus:

~~~python
parser.add_argument("--holiday-calendar", type=Path, required=True)
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=0.05)
parser.add_argument("--num-leaves", type=int, default=31)
parser.add_argument("--min-child-samples", type=int, default=20)
parser.add_argument("--evaluation-summary-path", default=None)
~~~

Keep output_directory as argparse's raw string (do not give that positional a Path type) so export_lightgbm_artifact can apply its shared raw-URI validator before any Path conversion. The supervised CSV and holiday-calendar inputs may be converted to Path locally after parsing.

Reuse the existing positive integer/float validation rules from the evaluation CLI. run_artifact_export must read the CSV once, calculate its raw-byte hash from the same path, parse the holiday calendar, construct LightGBMModelConfig with those four CLI tunables, and call export_lightgbm_artifact. main prints json.dumps of a JSON-safe manifest summary with sort_keys=True and returns:

~~~python
except (LightGBMArtifactCliError, SupervisedCsvError, LightGBMArtifactError, ModelTrainingError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 2
except LightGBMArtifactSerializationError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1
~~~

Create scripts/export_lightgbm_artifact.py as:

~~~python
from urbanflow.modeling.lightgbm_artifact_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
~~~

Do not import the rolling evaluator, MLflow, PostgreSQL, or FastAPI.

- [ ] **Step 4: Run focused verification**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/modeling/test_lightgbm_artifact.py tests/unit/modeling/test_lightgbm_artifact_cli.py -q
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/modeling/lightgbm_artifact_cli.py scripts/export_lightgbm_artifact.py tests/unit/modeling/test_lightgbm_artifact_cli.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/modeling/lightgbm_artifact_cli.py scripts/export_lightgbm_artifact.py tests/unit/modeling/test_lightgbm_artifact_cli.py
~~~

Expected: all artifact and CLI tests pass without filesystem output outside tmp_path.

- [ ] **Step 5: Commit the export-CLI task**

~~~powershell
git add src/urbanflow/modeling/lightgbm_artifact_cli.py scripts/export_lightgbm_artifact.py tests/unit/modeling/test_lightgbm_artifact_cli.py
git commit -m "feat(modeling): add lightgbm artifact export cli"
~~~

### Task 3: Add recent-history persistence reads and service-level forecast errors

**Files:**

- Modify: src/urbanflow/api/services.py
- Modify: src/urbanflow/api/errors.py
- Modify: src/urbanflow/api/postgres.py
- Modify: tests/unit/api/test_forecasts.py
- Modify: tests/unit/api/test_postgres_repositories.py

**Interfaces:**

- Consumes: existing HistoryRecord, DataStoreUnavailableError, PostgresSensorHistoryRepository, and standard UrbanFlowApiError handler.
- Produces:

~~~python
class RecentHistoryRepository(Protocol):
    def get_recent_history(
        self,
        location_id: int,
        *,
        limit: int,
    ) -> list[HistoryRecord]: ...

class ForecastInputUnavailableError(RuntimeError):
    pass

def forecast_unavailable_error() -> UrbanFlowApiError: ...
~~~

- PostgresSensorHistoryRepository additionally implements get_recent_history(location_id, *, limit).

- [ ] **Step 1: Write failing repository and error-mapping tests**

Extend the existing FakeSession tests. Add a deliberately descending list of PedestrianHourlyFact rows and assert this call returns them ascending:

~~~python
records = _repository(session).get_recent_history(101, limit=168)

assert [record.observed_at for record in records] == [
    earlier.observed_at,
    later.observed_at,
]
sql = _compile(session.statements[0])
assert "pedestrian_hourly_fact.location_id = 101" in sql
assert "ORDER BY pedestrian_hourly_fact.observed_at DESC" in sql
assert "LIMIT 168" in sql
assert session.closed
~~~

Add get_recent_history to every existing parameterized SQLAlchemy failure test and assert it raises DataStoreUnavailableError for factory, scalars, and row-consumption errors.

In test_forecasts.py add two providers:

~~~python
class UnavailableHistoryProvider:
    def predict(self, location_id: int, horizon: int) -> ForecastBatch:
        raise DataStoreUnavailableError("database unavailable")

class InvalidServingInputProvider:
    def predict(self, location_id: int, horizon: int) -> ForecastBatch:
        raise ForecastInputUnavailableError("missing contiguous history")
~~~

With an existing sensor and each provider configured, assert exact 503 bodies. The first preserves the existing data_store_unavailable message; the second is:

~~~python
{
    "error": {
        "code": "forecast_unavailable",
        "message": "Forecast cannot be generated from the available serving inputs.",
        "details": [],
    }
}
~~~

Keep the existing no-provider-before-sensor behavior and the configured-provider unknown-sensor behavior unchanged.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_postgres_repositories.py tests/unit/api/test_forecasts.py -q
~~~

Expected: imports or attribute lookups fail for RecentHistoryRepository, ForecastInputUnavailableError, forecast_unavailable_error, and get_recent_history.

- [ ] **Step 3: Implement the narrow persistence boundary and error conversion**

In services.py, define RecentHistoryRepository separately from HistoryRepository so existing history fakes do not gain a speculative method. Define ForecastInputUnavailableError beside DataStoreUnavailableError.

Wrap only the provider call in ForecastService.forecast:

~~~python
try:
    batch = self.model_provider.predict(location_id, horizon)
except DataStoreUnavailableError as exc:
    raise data_store_unavailable_error() from exc
except ForecastInputUnavailableError as exc:
    raise forecast_unavailable_error() from exc
~~~

Leave the existing model_provider is None check before _ensure_sensor_exists. Leave the existing complete-horizon, finite-value, ordering, and final non-negative clipping checks after the call.

Add forecast_unavailable_error in errors.py:

~~~python
def forecast_unavailable_error() -> UrbanFlowApiError:
    return UrbanFlowApiError(
        status_code=503,
        code="forecast_unavailable",
        message="Forecast cannot be generated from the available serving inputs.",
    )
~~~

In postgres.py add:

~~~python
def get_recent_history(
    self,
    location_id: int,
    *,
    limit: int,
) -> list[HistoryRecord]:
    statement = (
        select(PedestrianHourlyFact)
        .where(PedestrianHourlyFact.location_id == location_id)
        .order_by(PedestrianHourlyFact.observed_at.desc())
        .limit(limit)
    )
    try:
        with self._session_factory() as session:
            facts = session.scalars(statement).all()
    except SQLAlchemyError as exc:
        raise DataStoreUnavailableError("sensor data is unavailable") from exc
    return [
        HistoryRecord(
            observed_at=fact.observed_at,
            pedestrian_count=fact.pedestrian_count,
        )
        for fact in reversed(facts)
    ]
~~~

Do not add validation or fallback behavior to the repository; its caller owns the exact-168 serving contract.

- [ ] **Step 4: Run focused verification**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_postgres_repositories.py tests/unit/api/test_forecasts.py -q
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/api/services.py src/urbanflow/api/errors.py src/urbanflow/api/postgres.py tests/unit/api/test_postgres_repositories.py tests/unit/api/test_forecasts.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/api/services.py src/urbanflow/api/errors.py src/urbanflow/api/postgres.py tests/unit/api/test_postgres_repositories.py tests/unit/api/test_forecasts.py
~~~

Expected: the new query is bounded and descending in SQL but ascending at the port, and both provider exceptions become the standard JSON envelope.

- [ ] **Step 5: Commit the persistence/error task**

~~~powershell
git add src/urbanflow/api/services.py src/urbanflow/api/errors.py src/urbanflow/api/postgres.py tests/unit/api/test_forecasts.py tests/unit/api/test_postgres_repositories.py
git commit -m "feat(api): add forecast history boundary"
~~~

### Task 4: Implement strict artifact-backed direct LightGBM forecasting

**Files:**

- Create: src/urbanflow/api/lightgbm_provider.py
- Modify: src/urbanflow/api/services.py
- Modify: src/urbanflow/api/errors.py
- Create: tests/unit/api/test_lightgbm_provider.py
- Modify: tests/unit/api/test_forecasts.py

**Interfaces:**

- Consumes: LoadedLightGBMArtifact, RecentHistoryRepository, ForecastInputUnavailableError, ForecastBatch, ForecastPrediction, build_supervised_frame, UTC, and MELBOURNE_TZ.
- Produces:

~~~python
RECENT_HISTORY_LIMIT = 168

class ArtifactBackedLightGBMForecastProvider:
    def __init__(
        self,
        *,
        artifact: LoadedLightGBMArtifact,
        history_repository: RecentHistoryRepository,
    ) -> None: ...

    def predict(self, location_id: int, horizon: int) -> ForecastBatch: ...
~~~

- Define `ForecastModelOutputError` beside the existing provider-facing service exceptions. `ForecastService` catches it only around `model_provider.predict(...)` and maps it through a new error helper to `503 model_unavailable` with message `"Forecast provider returned an invalid prediction batch."`.

- The provider does not perform artifact I/O. Task 5 supplies an already validated LoadedLightGBMArtifact at startup.

- [ ] **Step 1: Write failing provider tests**

Create a RecordingRecentHistoryRepository with a records list, a calls list, and an optional DataStoreUnavailableError. Build an artifact under tmp_path using Task 1 with a valid all-weather-missing supervised CSV and a holiday calendar that covers the target dates. Cover this successful path:

~~~python
provider = ArtifactBackedLightGBMForecastProvider(
    artifact=loaded_artifact,
    history_repository=repository,
)

batch = provider.predict(location_id=101, horizon=24)

assert repository.calls == [(101, 168)]
assert batch.model_name == "lightgbm"
assert batch.model_version == loaded_artifact.manifest.model_version
assert batch.data_cutoff_at == history[-1].observed_at.astimezone(MELBOURNE_TZ)
assert batch.forecast_origin_at == batch.data_cutoff_at
assert [item.forecast_horizon for item in batch.predictions] == list(range(1, 25))
assert [item.target_at for item in batch.predictions] == [
    (batch.data_cutoff_at.astimezone(UTC) + timedelta(hours=step)).astimezone(
        MELBOURNE_TZ
    )
    for step in range(1, 25)
]
~~~

Use a small recording model only in the feature-semantics test. Its predict method receives the generated frame and returns one finite value per row. Assert:

1. the selected frame contains exactly horizons 1 through the requested horizon;
2. each forecast_origin_at equals the final observed cutoff;
3. is_public_holiday is true for a target date deliberately listed in the embedded calendar;
4. all weather columns are missing and their markers are true;
5. no predicted value was appended to the input history;
6. the provider normalizes a UTC record sequence crossing a Melbourne daylight-saving boundary to MELBOURNE_TZ while maintaining one-hour instant gaps.
7. target_at and holiday-calendar coverage advance by UTC instants through that daylight-saving transition, rather than by local wall-clock datetime addition.

Parameterize failure tests for 167 records, 169 records, reverse order, an exact one-hour gap, a naive timestamp, a minute=30 timestamp, a bool count, a float count, a negative count, and a calendar that does not cover the target day. Each must raise ForecastInputUnavailableError before model.predict is called. Separately cover True, 0, 25, and a non-integral horizon; each must fail before either repository or model is called. A repository DataStoreUnavailableError must propagate unchanged for ForecastService to map in Task 3.

Use recording models that return fewer and more prediction values than requested, plus values that cannot be converted to float. Call the provider through ForecastService with an existing sensor. None may become ForecastInputUnavailableError, be silently truncated, or fabricate a row; each must produce the exact `503 model_unavailable` code. Add the direct ForecastService error-envelope regression in test_forecasts.py for ForecastModelOutputError.

- [ ] **Step 2: Run the focused provider test and confirm RED**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_lightgbm_provider.py -q
~~~

Expected: collection fails because urbanflow.api.lightgbm_provider does not exist.

- [ ] **Step 3: Implement the provider with strict serving-input validation**

Create src/urbanflow/api/lightgbm_provider.py. Use explicit helpers rather than one large predict method:

~~~python
def _validate_horizon(horizon: int) -> None: ...
def _history_to_observations(
    location_id: int,
    records: list[HistoryRecord],
) -> pd.DataFrame: ...
def _validate_target_calendar(
    calendar: HolidayCalendar,
    *,
    cutoff: datetime,
    horizon: int,
) -> None: ...
def _forecast_rows(
    observations: pd.DataFrame,
    *,
    cutoff: datetime,
    horizon: int,
    calendar: HolidayCalendar,
) -> pd.DataFrame: ...
~~~

_validate_horizon must reject a bool, a non-Integral value, or a value outside 1 through 24 with ForecastInputUnavailableError. It is called before the repository read:

~~~python
if (
    isinstance(horizon, bool)
    or not isinstance(horizon, Integral)
    or horizon < 1
    or horizon > 24
):
    raise ForecastInputUnavailableError("forecast horizon is invalid")
~~~

_history_to_observations must require len(records) == RECENT_HISTORY_LIMIT. For each record:

~~~python
if observed_at.tzinfo is None or observed_at.utcoffset() is None:
    raise ForecastInputUnavailableError("history timestamp is timezone-naive")
instant = observed_at.astimezone(UTC)
if any((instant.minute, instant.second, instant.microsecond)):
    raise ForecastInputUnavailableError("history timestamp is not an exact hour")
if previous_instant is not None and instant - previous_instant != timedelta(hours=1):
    raise ForecastInputUnavailableError("history is not contiguous")
if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
    raise ForecastInputUnavailableError("history count is invalid")
~~~

Store each accepted instant as instant.astimezone(MELBOURNE_TZ), preserving both the source instant and Melbourne-local feature semantics. Build a DataFrame containing only location_id, observed_at, and pedestrian_count; do not insert weather values or forecast targets.

_validate_target_calendar checks each local date reached by advancing the cutoff's UTC instant one hour at a time, then converting to MELBOURNE_TZ. It must not use `cutoff + timedelta(...)`, whose wall-clock semantics can skip a real instant at a daylight-saving fallback:

~~~python
target_at = (
    cutoff.astimezone(UTC) + timedelta(hours=step)
).astimezone(MELBOURNE_TZ)
if not calendar.contains(target_at.date()):
    raise ForecastInputUnavailableError("holiday calendar does not cover target date")
~~~

_forecast_rows calls:

~~~python
supervised = build_supervised_frame(
    observations,
    horizons=range(1, horizon + 1),
    public_holidays=calendar.public_holidays,
)
rows = supervised.loc[
    supervised["forecast_origin_at"] == pd.Timestamp(cutoff)
].sort_values("forecast_horizon")
if rows["forecast_horizon"].tolist() != list(range(1, horizon + 1)):
    raise ForecastInputUnavailableError("could not construct direct forecast rows")
~~~

predict validates horizon first, then obtains the records once with limit=168, calls these helpers, invokes artifact.model.predict(rows) once, and constructs ForecastPrediction values in forecast_horizon order from target_observed_at. Set generated_at=datetime.now(UTC), data_cutoff_at=cutoff, forecast_origin_at=cutoff, model_name=lightgbm, and model_version=artifact.manifest.model_version. Convert outputs to floats; if conversion fails or their count differs from the selected rows in either direction, raise ForecastModelOutputError. ForecastService maps that exception to the specified `model_unavailable` response; do not use ForecastInputUnavailableError, silently truncate output, or fabricate a row. Non-finite float values may reach the existing service-level finite-value validation. Do not catch DataStoreUnavailableError; do not clip predictions here; do not mutate the repository result.

- [ ] **Step 4: Run focused verification**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_lightgbm_provider.py tests/unit/api/test_forecasts.py -q
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/api/lightgbm_provider.py tests/unit/api/test_lightgbm_provider.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/api/lightgbm_provider.py tests/unit/api/test_lightgbm_provider.py
~~~

Expected: real temporary artifact predictions form one direct batch; invalid runtime history never reaches model prediction; existing route-level clipping and malformed-batch tests remain green.

- [ ] **Step 5: Commit the provider task**

~~~powershell
git add src/urbanflow/api/lightgbm_provider.py src/urbanflow/api/services.py src/urbanflow/api/errors.py tests/unit/api/test_lightgbm_provider.py tests/unit/api/test_forecasts.py
git commit -m "feat(api): add lightgbm artifact forecast provider"
~~~

### Task 5: Wire the real provider only for a valid database-and-artifact configuration

**Files:**

- Modify: src/urbanflow/api/app.py
- Modify: tests/conftest.py
- Modify: tests/unit/api/test_app.py

**Interfaces:**

- Consumes: DATABASE_URL_ENV_VAR, PostgresSensorHistoryRepository, load_lightgbm_artifact, LightGBMArtifactError, ArtifactBackedLightGBMForecastProvider, and ApiServices.
- Produces:

~~~python
MODEL_ARTIFACT_PATH_ENV_VAR = "URBANFLOW_API_MODEL_ARTIFACT_PATH"

def create_default_services(
    *,
    environ: Mapping[str, str] | None = None,
) -> ApiServices: ...
~~~

- Explicit create_app(services=injected_services) continues to bypass all environment inspection, database construction, and artifact loading.

- [ ] **Step 1: Write failing configuration-matrix tests**

In tests/conftest.py remove both DATABASE_URL_ENV_VAR and MODEL_ARTIFACT_PATH_ENV_VAR before test modules import the module-level Uvicorn app.

In test_app.py use monkeypatch to replace create_database_engine, create_session_factory, and load_lightgbm_artifact. Add one test per row:

| Input environment | Required assertion |
| --- | --- |
| neither setting or whitespace-only database URL | EmptySensorRepository, EmptyHistoryRepository, model_provider is None, and no loader call |
| valid database URL only | one lazy PostgresSensorHistoryRepository in both repository slots, model_provider is None, and no loader call |
| artifact directory only | empty repositories, model_provider is None, no Engine/session/loader call |
| valid database URL plus valid artifact path | one shared PostgreSQL repository and an ArtifactBackedLightGBMForecastProvider constructed with it |
| valid database URL plus loader LightGBMArtifactError | PostgreSQL repositories remain installed, model_provider is None, and sensor/history reads still reach the repository |

For the invalid-artifact row, use the existing StatementAwareSessionFactory and assert GET /api/v1/sensors succeeds while GET /api/v1/sensors/999001/forecast?horizon=1 returns the existing exact model_unavailable response. Also assert a malformed database URL still raises DatabaseConfigError, and injected ApiServices is used unchanged even when both environment strings are invalid.

- [ ] **Step 2: Run the focused app tests and confirm RED**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_app.py -q
~~~

Expected: the new artifact-path constant/imports and matrix behavior fail.

- [ ] **Step 3: Implement opt-in default-service wiring**

In app.py define MODEL_ARTIFACT_PATH_ENV_VAR next to the factory or import one canonical constant from lightgbm_artifact. Resolve both strings from the supplied environ mapping, trimming whitespace. Keep the existing early return when the database URL is absent or blank:

~~~python
if configured_database_url is None or not configured_database_url.strip():
    return ApiServices()
~~~

Only after the valid lazy PostgreSQL repository exists, decide whether to load a model:

~~~python
configured_artifact_path = values.get(MODEL_ARTIFACT_PATH_ENV_VAR)
model_provider: ForecastModelProvider | None = None
if configured_artifact_path is not None and configured_artifact_path.strip():
    try:
        artifact = load_lightgbm_artifact(configured_artifact_path.strip())
    except LightGBMArtifactError:
        model_provider = None
    else:
        model_provider = ArtifactBackedLightGBMForecastProvider(
            artifact=artifact,
            history_repository=repository,
        )

return ApiServices(
    sensor_repository=repository,
    history_repository=repository,
    model_provider=model_provider,
)
~~~

Do not catch DatabaseConfigError or SQLAlchemy ArgumentError beyond the existing malformed-URL conversion. Do not open a session, probe the model, call predict, change HealthService, or make artifact errors a startup exception.

- [ ] **Step 4: Run focused verification**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_app.py tests/unit/api/test_sensors.py tests/unit/api/test_history.py tests/unit/api/test_forecasts.py -q
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/api/app.py tests/conftest.py tests/unit/api/test_app.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/api/app.py tests/conftest.py tests/unit/api/test_app.py
~~~

Expected: configuration is lazy and deterministic; artifact-only never triggers I/O; bad artifacts degrade forecast only; injected tests remain isolated from ambient environment variables.

- [ ] **Step 5: Commit the app-wiring task**

~~~powershell
git add src/urbanflow/api/app.py tests/conftest.py tests/unit/api/test_app.py
git commit -m "feat(api): wire lightgbm forecast artifacts"
~~~

### Task 6: Add an opt-in real PostgreSQL-and-artifact provider smoke

**Files:**

- Create: src/urbanflow/api/lightgbm_forecast_smoke.py
- Create: scripts/smoke_test_lightgbm_forecast.py
- Create: tests/unit/api/test_lightgbm_forecast_smoke.py

**Interfaces:**

- Consumes: create_database_engine, Base metadata, repository upsert helpers, PostgresSensorHistoryRepository, artifact export/load, and ArtifactBackedLightGBMForecastProvider.
- Produces:

~~~python
SMOKE_DATABASE_URL_ENV_VAR = "URBANFLOW_SMOKE_DATABASE_URL"

@dataclass(frozen=True)
class LightGBMForecastSmokeResult:
    schema_name: str
    location_id: int
    data_cutoff_at: str
    forecast_horizons: list[int]
    model_version: str

def run_lightgbm_forecast_smoke(
    database_url: str,
    *,
    schema_name: str | None = None,
) -> LightGBMForecastSmokeResult: ...
def main(argv: list[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int: ...
~~~

- [ ] **Step 1: Write database-free smoke validation tests**

Copy only the safe schema-name and missing-URL test style from test_postgres_smoke.py. Assert:

1. validate_smoke_schema_name accepts generated lowercase names and rejects dangerous identifiers;
2. main returns 2 and names URBANFLOW_SMOKE_DATABASE_URL when neither a flag nor environment supplies a database URL;
3. parser accepts an optional --schema-name and no artifact/network arguments;
4. LightGBMForecastSmokeResult serializes with JSON-safe primitive values, including an ISO-8601 string data_cutoff_at.

Do not run run_lightgbm_forecast_smoke in routine pytest because it deliberately requires a PostgreSQL server.

- [ ] **Step 2: Run the focused smoke test and confirm RED**

Run:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_lightgbm_forecast_smoke.py -q
~~~

Expected: collection fails because lightgbm_forecast_smoke does not exist.

- [ ] **Step 3: Implement the bounded opt-in integration smoke**

Follow the existing postgres_smoke.py lifecycle exactly: validate or generate a safe schema name before creating an engine, create only that schema, set its search_path on the smoke connection, create tables, and always DROP SCHEMA IF EXISTS quoted_schema CASCADE in finally after successful schema creation and before engine.dispose. Use only the validated, quoted schema identifier for CREATE, SET search_path, and DROP; bind the smoke session factory to the same configured Connection rather than the Engine.

Inside tempfile.TemporaryDirectory:

1. create a Melbourne-local 192-hour deterministic history for location 999001 and its one-row sensor;
2. build supervised rows with build_supervised_frame and a small explicit holiday calendar;
3. write the CSV and calendar under the temporary directory;
4. export and load a five-tree artifact using LightGBMModelConfig(n_estimators=5, min_child_samples=1);
5. upsert the sensor and all hourly rows into the temporary schema;
6. construct PostgresSensorHistoryRepository from that same schema-bound session factory, then construct ArtifactBackedLightGBMForecastProvider;
7. call predict(999001, 24) and assert horizons 1 through 24, finite non-negative predictions, a cutoff equal to the final source instant, and the artifact model version.

Return only schema_name, location_id, ISO-safe cutoff data, ordered horizons, and model version; data_cutoff_at is `batch.data_cutoff_at.isoformat()` so `json.dumps(asdict(result), sort_keys=True)` succeeds without a custom encoder. main prints that JSON, returns 2 for configuration/value errors, and 1 for SQLAlchemy or artifact/provider operational failures. Catch artifact/model/provider/SQLAlchemy failures before a generic ValueError branch because LightGBMArtifactError is a ValueError subclass: LightGBMArtifactError, LightGBMArtifactSerializationError, ModelTrainingError, DataStoreUnavailableError, ForecastInputUnavailableError, ForecastModelOutputError, and SQLAlchemyError return 1; missing URL, invalid schema, and other configuration/value errors return 2. The script wrapper contains only:

~~~python
from urbanflow.api.lightgbm_forecast_smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
~~~

- [ ] **Step 4: Run focused verification and, only if intentionally configured, the live smoke**

Run the database-free checks:

~~~powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/api/test_lightgbm_forecast_smoke.py -q
& .\.venv\Scripts\python.exe -m ruff check src/urbanflow/api/lightgbm_forecast_smoke.py scripts/smoke_test_lightgbm_forecast.py tests/unit/api/test_lightgbm_forecast_smoke.py
& .\.venv\Scripts\python.exe -m ruff format --check src/urbanflow/api/lightgbm_forecast_smoke.py scripts/smoke_test_lightgbm_forecast.py tests/unit/api/test_lightgbm_forecast_smoke.py
~~~

Only when a local disposable PostgreSQL database is intentionally available, run:

~~~powershell
$env:URBANFLOW_SMOKE_DATABASE_URL = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"
& .\.venv\Scripts\python.exe scripts/smoke_test_lightgbm_forecast.py
~~~

Expected manual result: JSON with location_id 999001, forecast_horizons [1, ..., 24], a nonempty model_version, and no persistent schema or model artifact after completion.

- [ ] **Step 5: Commit the optional live-smoke task**

~~~powershell
git add src/urbanflow/api/lightgbm_forecast_smoke.py scripts/smoke_test_lightgbm_forecast.py tests/unit/api/test_lightgbm_forecast_smoke.py
git commit -m "test(api): add lightgbm forecast smoke"
~~~

### Task 7: Synchronize operator documentation and CI health smoke

**Files:**

- Modify: README.md
- Modify: urbanflow-au_requirements.md
- Modify: .github/workflows/ci.yml
- Modify: docs/superpowers/plans/2026-07-16-lightgbm-artifact-forecast-serving.md

**Interfaces:**

- Consumes: final exporter CLI, URBANFLOW_DATABASE_URL, URBANFLOW_API_MODEL_ARTIFACT_PATH, the existing API route contract, and the optional smoke script.
- Produces: accurate local serving instructions, an explicit non-production boundary, and one CI-managed default Uvicorn health check.

- [ ] **Step 1: Update README and requirements facts**

In README.md:

1. replace the current statement that model artifacts and real provider are future work with the precise condition: real forecasts exist only when both URBANFLOW_DATABASE_URL and a valid operator-controlled URBANFLOW_API_MODEL_ARTIFACT_PATH are configured;
2. add the exact export example:

~~~powershell
python scripts/export_lightgbm_artifact.py data/modeling/supervised_rows.csv models/lightgbm/local-demo --holiday-calendar data/modeling/holiday_calendar.json
~~~

3. state that the artifact directory is ignored by Git, contains model.joblib plus manifest.json, is trusted local operator input only, and is not an MLflow Registry artifact;
4. document the required holiday-calendar JSON shape and that serving returns 503 forecast_unavailable when its requested target dates are outside calendar coverage;
5. state that this first artifact slice intentionally rejects training data with observed weather values because it has no weather serving source;
6. add a Uvicorn example setting both variables, then a forecast Invoke-RestMethod call;
7. retain the safe-default statements: missing database means no Engine or artifact read; database-only or invalid artifact retains sensor/history but forecast 503 model_unavailable; no prediction is fabricated;
8. list scripts/smoke_test_lightgbm_forecast.py as an opt-in disposable PostgreSQL integration check, not a routine test;
9. keep the boundary explicit: no registry, retraining, Dashboard, monitoring, Docker, or production-performance claim is introduced.

In urbanflow-au_requirements.md section 10, replace the claim that model artifact loading and the real provider are future work with the same two-variable contract, artifact provenance/feature constraints, direct 1–24 horizon semantics, and the three forecast failure categories:

| Condition | HTTP result |
| --- | --- |
| no valid provider | 503 model_unavailable |
| storage unavailable | 503 data_store_unavailable |
| invalid current history or holiday coverage | 503 forecast_unavailable |

Do not change the existing route paths, route versions, dashboard roadmap, or health specification.

- [ ] **Step 2: Add a bounded CI Uvicorn health smoke**

After the pytest step in .github/workflows/ci.yml, add exactly one step that starts only the default no-environment app and always stops its own process:

~~~yaml
      - name: Smoke default Uvicorn health endpoint
        run: |
          set -eu
          python -m uvicorn urbanflow.api.app:app --host 127.0.0.1 --port 8765 > uvicorn.log 2>&1 &
          server_pid=$!
          cleanup() {
            kill "$server_pid" 2>/dev/null || true
            wait "$server_pid" 2>/dev/null || true
          }
          trap cleanup EXIT
          ready=0
          for attempt in $(seq 1 80); do
            if python -c "import json, urllib.request; body = json.load(urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=1)); assert body['status'] == 'degraded'"; then
              ready=1
              break
            fi
            sleep 0.25
          done
          if [ "$ready" -ne 1 ]; then
            cat uvicorn.log
            exit 1
          fi
~~~

This uses a 20-second readiness bound, does not start PostgreSQL, does not set an artifact path, and checks the existing default degraded status without changing health semantics.

- [ ] **Step 3: Mark only completed plan checkboxes and run documentation checks**

After Tasks 1 through 6 have actually passed, change only their completed checkboxes in this plan from [ ] to [x]. Do not pre-mark Task 8. Then run:

~~~powershell
$ErrorActionPreference = "Stop"
git diff --check
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest -q
~~~

Expected: documentation is internally consistent, the CI YAML is valid by GitHub Actions parsing on push, and the complete offline suite passes.

- [ ] **Step 4: Commit documentation and CI**

~~~powershell
git add README.md urbanflow-au_requirements.md .github/workflows/ci.yml docs/superpowers/plans/2026-07-16-lightgbm-artifact-forecast-serving.md
git commit -m "docs: describe artifact-backed forecasts"
~~~

### Task 8: Review, verify, integrate, and confirm CI without unbounded waits

**Files:**

- Verify all files from Tasks 1 through 7; do not add new behavior except a regression fix backed by a focused test.

- [ ] **Step 1: Review the completed feature branch against the approved design**

Review main...HEAD against docs/superpowers/specs/2026-07-16-lightgbm-artifact-forecast-serving-design.md. Reject or repair any Critical, P1, or P2 issue before integration. Confirm all of the following:

1. model.joblib holds the fitted wrapper/pipeline, not only a booster;
2. no configured database means no artifact loader invocation;
3. invalid artifacts cannot break sensor/history reads or invent forecasts;
4. history checks happen before feature building and all direct predictions share one cutoff;
5. timezone, holiday, weather-missing, feature-order, and checksum checks have focused regressions;
6. no health, dashboard, migration, registry, Docker, or network behavior slipped into the diff.

- [ ] **Step 2: Run the feature-branch full quality gate**

Run:

~~~powershell
$ErrorActionPreference = "Stop"
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest -q
git status --short --branch
~~~

Expected: zero Ruff issues, all tests pass offline, and the only branch is clean codex/lightgbm-artifact-forecast-serving.

- [ ] **Step 3: Run a bounded local Uvicorn smoke**

Run this from the feature worktree. It uses a random loopback port, starts a hidden child process, waits at most 20 seconds, and stops only that child:

~~~powershell
$ErrorActionPreference = "Stop"
$stdout = New-TemporaryFile
$stderr = New-TemporaryFile
$process = $null
try {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    $listener.Stop()
    $process = Start-Process -FilePath .\.venv\Scripts\python.exe -ArgumentList @("-m", "uvicorn", "urbanflow.api.app:app", "--host", "127.0.0.1", "--port", "$port") -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $deadline = (Get-Date).AddSeconds(20)
    $health = $null
    while ((Get-Date) -lt $deadline -and $null -eq $health) {
        try {
            $candidate = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2
            if ($candidate.status -eq "degraded") { $health = $candidate }
        } catch {}
        if ($null -eq $health) { Start-Sleep -Milliseconds 250 }
    }
    if ($null -eq $health) {
        throw "Uvicorn did not become healthy within 20 seconds: $(Get-Content -Raw $stderr)"
    }
    $paths = (Invoke-RestMethod -Uri "http://127.0.0.1:$port/openapi.json" -TimeoutSec 2).paths.PSObject.Properties.Name
    $expected = @(
        "/health",
        "/api/v1/sensors",
        "/api/v1/sensors/{location_id}/history",
        "/api/v1/sensors/{location_id}/forecast",
        "/api/v1/model/metrics"
    )
    if (Compare-Object $expected @($paths)) { throw "OpenAPI route set changed." }
} finally {
    if ($null -ne $process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
    Remove-Item $stdout, $stderr -ErrorAction SilentlyContinue
}
~~~

Expected: health is 200/degraded and the exact five existing routes remain mounted. This smoke does not use a database or artifact.

- [ ] **Step 4: Synchronize main and rebase with bounded Git network operations**

Before any fetch, verify both worktrees:

~~~powershell
$rootWorktree = "D:\Github项目\UrbanFlow-AU"
$featureWorktree = "D:\Github项目\UrbanFlow-AU\.worktrees\lightgbm-artifact-forecast-serving"
if ((git -C $rootWorktree branch --show-current).Trim() -ne "main") {
    throw "Root worktree must be on main."
}
if (git -C $rootWorktree status --porcelain) {
    throw "Root worktree is not clean."
}
if ((git -C $featureWorktree branch --show-current).Trim() -ne "codex/lightgbm-artifact-forecast-serving") {
    throw "Feature worktree branch is incorrect."
}
if (git -C $featureWorktree status --porcelain) {
    throw "Feature worktree is not clean."
}
~~~

Use this bounded helper rather than an unbounded native Git command:

~~~powershell
function Invoke-BoundedNative {
    param([string]$FilePath, [string[]]$ArgumentList, [int]$TimeoutSeconds = 60)
    $stdout = New-TemporaryFile
    $stderr = New-TemporaryFile
    $process = $null
    try {
        $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-Process -Id $process.Id -Force
            throw "$FilePath exceeded $TimeoutSeconds seconds."
        }
        $out = Get-Content -Raw $stdout
        $err = Get-Content -Raw $stderr
        if ($process.ExitCode -ne 0) {
            throw "$FilePath failed with exit code $($process.ExitCode): $err"
        }
        return $out
    } finally {
        Remove-Item $stdout, $stderr -ErrorAction SilentlyContinue
    }
}

try {
    Invoke-BoundedNative git @("-C", $rootWorktree, "fetch", "origin") 60 | Out-Host
} catch {
    if ($_.Exception.Message -notmatch "exceeded|Could not resolve|Failed to connect|Connection timed out|Network is unreachable") {
        throw
    }
    Invoke-BoundedNative git @("-c", "http.proxy=http://127.0.0.1:10808", "-C", $rootWorktree, "fetch", "origin") 60 | Out-Host
}
git -C $rootWorktree merge --ff-only origin/main
git -C $featureWorktree rebase main
~~~

Expected: main equals origin/main and the clean feature branch is rebased. The proxy is one command only; do not persist a Git proxy configuration. If the remote advanced or a conflict appears, do not force push or merge; resolve with a focused test and repeat Steps 1 through 3.

- [ ] **Step 5: Re-run the rebased branch gate, fast-forward main, push only main, and verify Actions**

Run the Task 8 Step 2 gate again in the rebased feature worktree. Then:

~~~powershell
git -C $rootWorktree merge --ff-only codex/lightgbm-artifact-forecast-serving
Set-Location $rootWorktree
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest -q
Invoke-BoundedNative git @("-C", $rootWorktree, "push", "origin", "main") 60 | Out-Host
~~~

If the direct push has a connection-only failure, repeat just the bounded push with the same one-command http.proxy fallback. A non-fast-forward response returns to Step 4; never force push.

Read the public GitHub Actions API for the exact local HEAD SHA with bounded requests, poll at most 15 minutes, and require a completed success conclusion before cleanup. Use this helper and loop; it reports every poll result and restores any temporary proxy variables:

~~~powershell
function Invoke-GitHubApi {
    param([string]$Uri)
    try {
        return Invoke-RestMethod -Uri $Uri -Headers @{ "User-Agent" = "UrbanFlow-AU-Codex" } -TimeoutSec 30
    } catch {
        if ($_.Exception.Message -notmatch "Could not resolve|Failed to connect|Connection timed out|Network is unreachable") {
            throw
        }
        $hadHttpsProxy = Test-Path Env:HTTPS_PROXY
        $previousHttpsProxy = $env:HTTPS_PROXY
        try {
            $env:HTTPS_PROXY = "http://127.0.0.1:10808"
            return Invoke-RestMethod -Uri $Uri -Headers @{ "User-Agent" = "UrbanFlow-AU-Codex" } -TimeoutSec 30
        } finally {
            if ($hadHttpsProxy) { $env:HTTPS_PROXY = $previousHttpsProxy }
            else { Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue }
        }
    }
}

$head = (git -C $rootWorktree rev-parse HEAD).Trim()
$discoveryDeadline = (Get-Date).AddMinutes(5)
$run = $null
while ((Get-Date) -lt $discoveryDeadline -and $null -eq $run) {
    $runs = Invoke-GitHubApi "https://api.github.com/repos/PoorJeff/UrbanFlow-AU/actions/runs?branch=main&event=push&per_page=20"
    $run = @($runs.workflow_runs | Where-Object { $_.head_sha -eq $head } | Select-Object -First 1)[0]
    if ($null -eq $run) {
        Write-Output "Waiting for GitHub Actions run for $head"
        Start-Sleep -Seconds 5
    }
}
if ($null -eq $run) { throw "No Actions run appeared for $head within five minutes." }
$completionDeadline = (Get-Date).AddMinutes(15)
do {
    $status = Invoke-GitHubApi "https://api.github.com/repos/PoorJeff/UrbanFlow-AU/actions/runs/$($run.id)"
    Write-Output "GitHub Actions $($run.id): status=$($status.status), conclusion=$($status.conclusion)"
    if ($status.status -ne "completed") { Start-Sleep -Seconds 5 }
} while ($status.status -ne "completed" -and (Get-Date) -lt $completionDeadline)
if ($status.status -ne "completed" -or $status.conclusion -ne "success") {
    throw "GitHub Actions $($run.id) did not conclude successfully."
}
~~~

After success only:

~~~powershell
git -C $rootWorktree worktree remove $featureWorktree
git -C $rootWorktree worktree prune
git -C $rootWorktree branch -d codex/lightgbm-artifact-forecast-serving
~~~

Expected: main is the sole pushed branch, its complete quality gate and exact-head CI run succeed, and only then is the local feature worktree removed.

## Plan Self-Review

### Spec coverage

- Task 1 covers the closed artifact schema, direct joblib dependency, full preprocessing pipeline, local-only atomic bundle, model/source hashes, feature/config checks, final fitting, weather compatibility, and explicit holiday-calendar format.
- Task 2 covers the runnable exporter contract and its precise success, invalid-input, and serialization exit codes.
- Task 3 covers the narrow recent-history port, descending bounded SQL query, ascending API records, and standard data-store/forecast-input errors without changing current no-provider precedence.
- Task 4 covers exact 168-history semantics, UTC continuity, Melbourne normalization, daylight saving, holiday coverage, all-missing weather feature parity, direct multi-horizon generation, provider metadata, and no-recursion behavior.
- Task 5 covers the entire environment configuration matrix, lazy artifact loading, invalid-artifact degradation, and injected-service precedence.
- Task 6 supplies the explicitly opt-in real PostgreSQL plus temporary artifact evidence while retaining fully offline routine tests.
- Task 7 updates all user-facing claims and adds a bounded default Uvicorn smoke to CI without changing health behavior.
- Task 8 covers independent review, full gates, local smoke, bounded network operations, fast-forward-only integration, exact-head CI verification, and post-success cleanup.

### Placeholder scan

The plan has no unresolved placeholders, unspecified validation, or implicit test work. Every code-bearing task names its files, produces concrete interfaces, states its RED and GREEN commands, and defines an intentional commit.

### Type consistency

LoadedLightGBMArtifact contains the FittedLightGBMModel used by ArtifactBackedLightGBMForecastProvider. The provider consumes RecentHistoryRepository and produces the unchanged ForecastBatch protocol. PostgresSensorHistoryRepository implements that new port while remaining both the existing SensorRepository and HistoryRepository. create_default_services passes that one repository instance into ApiServices and the provider only when both explicit settings are valid. ForecastService remains the sole location that maps provider DataStoreUnavailableError and ForecastInputUnavailableError to the existing standard FastAPI JSON envelope.

## Execution Handoff

Plan complete and saved to docs/superpowers/plans/2026-07-16-lightgbm-artifact-forecast-serving.md. Two execution options:

1. Subagent-Driven (recommended) - dispatch a fresh subagent per task, review between tasks, and keep each task independently reversible.
2. Inline Execution - execute tasks in this session in small batches with explicit verification checkpoints.

Choose one approach before implementation begins.
