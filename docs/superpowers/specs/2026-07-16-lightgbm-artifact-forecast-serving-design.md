# LightGBM Artifact Forecast Serving Design

**Date:** 2026-07-16

## Goal

Turn the existing `GET /api/v1/sensors/{location_id}/forecast` contract into
an honest, local, artifact-backed 1–24 hour LightGBM forecast path. The
provider must use persisted PostgreSQL history, retain the project's direct
multi-horizon feature semantics, and remain fully opt-in.

## Current context

UrbanFlow AU already has:

- leakage-safe supervised features with direct horizons `1..24`;
- fitted Ridge and LightGBM Python model wrappers;
- batch evaluation CLIs and MLflow evidence tracking;
- a typed FastAPI forecast route with an injectable `ForecastModelProvider`;
- optional PostgreSQL sensor/history reads through
  `PostgresSensorHistoryRepository`.

The forecast route still returns `503 model_unavailable` by default because no
model artifact or real provider exists. PostgreSQL reads now make it possible
to derive serving features from actual recent history, so a model artifact and
provider are the next smallest vertical slice.

## Decision

Implement **one explicit LightGBM artifact format and one LightGBM serving
provider**. Do not support Ridge selection, MLflow Model Registry selection,
remote artifact downloads, automatic retraining, Dashboard work, monitoring,
or Docker Compose in this slice.

LightGBM is selected because it is the project's intended global model and
already has a fitted wrapper that owns the preprocessing pipeline. Supporting
both models now would duplicate artifact, loading, configuration, test, and
documentation paths without improving the first runnable product flow.

## Scope

### Included

1. Export an explicit, local LightGBM artifact bundle from an existing
   supervised CSV.
2. Validate and load that bundle only when an explicit environment variable
   configures it.
3. Read the latest 168 hourly observations for a sensor from PostgreSQL.
4. Reject insufficient, non-hourly, or gapped serving history instead of
   allowing model imputers to turn it into an unexplained forecast.
5. Reuse `build_supervised_frame(...)` to create direct `1..24` serving rows
   at the real data cutoff.
6. Produce a real `ForecastBatch` through the existing API provider protocol.
7. Preserve the safe default: no database URL and no artifact path means no
   engine, no database connection, no artifact I/O, and `503
   model_unavailable` for forecasts.

### Explicit non-goals

- Ridge artifact export or model selection;
- MLflow Model Registry, remote tracking server, or remote artifact download;
- automatic artifact promotion, retraining, scheduling, or database-native
  training/evaluation reads;
- weather forecast ingestion, public-holiday downloading, or use of future
  observed weather;
- PostgreSQL migrations or table changes;
- Streamlit, Plotly, Evidently, Docker Compose, or dashboard changes;
- a fallback that invents or recursively feeds predictions as lag values;
- health endpoint probing or changed health semantics.

## Artifact contract

`URBANFLOW_API_MODEL_ARTIFACT_PATH` names a local artifact **directory**. It
must never be interpreted as a URL. The directory is normally created below
the ignored `models/` tree and contains exactly:

```text
<artifact-directory>/
├── manifest.json
└── model.joblib
```

The raw configured string is checked before `Path` conversion: a value with
`://` is rejected. The local-path validator also rejects an already-normalized
URI-like form such as `s3:\bucket`, which Windows can produce from
`Path("s3://bucket")`, while accepting ordinary drive paths such as
`C:\models\artifact`. This same validator is used by export, load, and
environment-driven API construction, so a remote-looking value can never
become a local relative path by accident.

`model.joblib` is a `FittedLightGBMModel`, including the fitted scikit-learn
preprocessing pipeline. `manifest.json` has this versioned, JSON-only shape:

```json
{
  "schema_version": 1,
  "model_name": "lightgbm",
  "model_version": "lightgbm-<first-12-sha256-hex>",
  "model_sha256": "<64-lowercase-hex>",
  "training_data_sha256": "<64-lowercase-hex>",
  "created_at": "2026-07-16T00:00:00+00:00",
  "trained_through_at": "2026-07-15T23:00:00+00:00",
  "training_row_count": 1234,
  "feature_timezone": "Australia/Melbourne",
  "feature_columns": ["forecast_horizon", "pedestrian_count"],
  "model_config": {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "random_state": 42
  },
  "holiday_calendar_start": "2026-01-01",
  "holiday_calendar_end": "2026-12-31",
  "public_holidays": ["2026-01-26"],
  "evaluation_summary_path": null
}
```

The actual exported `feature_columns` list is the complete ordered
`FittedLightGBMModel.feature_columns`, not the abbreviated example above.
`trained_through_at` is the maximum timezone-aware `forecast_origin_at` among
rows used to fit the artifact; it is distinct from each request's live
`data_cutoff_at`.

Schema version `1` has a deliberately closed contract. `schema_version` is the
integer `1`; `model_name` is exactly `"lightgbm"`; both SHA-256 fields are
64 lowercase hexadecimal characters; and `model_version` is exactly
`lightgbm-` plus the first twelve characters of `model_sha256`. `created_at`
and `trained_through_at` are offset-bearing ISO-8601 timestamps,
`training_row_count` is a positive non-boolean integer, and
`feature_timezone` is exactly `Australia/Melbourne`. The ordered manifest
feature columns must match both the deserialized model's `feature_columns` and
its `config.feature_spec.feature_columns`; the serving slice supports only the
project's existing default feature specification. The six scalar model-config
values must equal the deserialized `LightGBMModelConfig` values:
`n_estimators`, `num_leaves`, and `min_child_samples` are positive integers;
`learning_rate` is finite and positive; `max_depth` is an integer not below
`-1`; and `random_state` is an integer. `training_data_sha256` is the SHA-256
of the exact source CSV bytes, retained for local provenance rather than as an
artifact-version substitute.

The embedded holiday calendar is local, explicit, and self-contained:
`holiday_calendar_start` and `holiday_calendar_end` are inclusive ISO dates;
the start is not after the end; and `public_holidays` is a sorted, duplicate-free
list of ISO dates contained by that coverage. It is not inferred from the
supervised CSV and is never downloaded. A forecast is available only when all
of its future target dates are within this coverage. This makes an exhausted
calendar a truthful service-input failure rather than silently marking unknown
future holidays as ordinary days.

The exporter serializes the model to a temporary sibling directory, calculates
the SHA-256 of `model.joblib`, writes the manifest, then renames the directory
into the requested new destination. It refuses an existing destination rather
than overwriting an artifact. This prevents a partial bundle from appearing at
the configured path.

The loader validates the complete manifest contract above, exact model name,
model checksum, model-version derivation, artifact schema version, ordered
feature columns, timezone, calendar coverage, and model configuration before
accepting the deserialized object. `joblib` is a direct runtime dependency and
artifacts are only safe when their containing directory is operator-controlled;
the service must never load an artifact obtained from an untrusted source.

## Export path

Add a CLI such as:

```powershell
python scripts/export_lightgbm_artifact.py `
  data/modeling/supervised_rows.csv `
  models/lightgbm/local-demo `
  --holiday-calendar data/modeling/holiday_calendar.json
```

The command reads the existing supervised CSV, parses its offset-bearing
timestamp columns,
fits one final `FittedLightGBMModel` using every non-missing target row in that
operator-supplied CSV, and writes the bundle described above. It does not reuse
the rolling evaluator or claim that its final fit is a new evaluation. Its
LightGBM arguments match the existing evaluator's
parameters (`n_estimators`, `learning_rate`, `num_leaves`, and
`min_child_samples`) so the export configuration is explicit and reproducible.
It returns `2` for invalid input or output paths and `1` for serialization
failures. It does not call MLflow, contact a database, or download data.

The reader rejects a timestamp with no timezone offset. For a CSV that spans
Melbourne's daylight-saving change, it validates each source value before
normalizing the series to timezone-aware UTC instants; it must not silently
treat a naive timestamp as UTC. This preserves chronology even when adjacent
source rows use both `+10:00` and `+11:00` offsets.

`--holiday-calendar` points to a local JSON object with exactly
`coverage_start`, `coverage_end`, and `public_holidays` keys; the first two are
inclusive ISO dates and the last is an array of ISO dates. The exporter copies
the validated calendar into the manifest. Since this serving slice has no
weather source, it also rejects a training frame in which an eligible row has
an observed `temperature`, `rainfall`, or `wind_speed` value, or marks one of
those fields as not missing. Consequently, the fitted artifact and serving
feature builder both use the existing explicit-weather-missing behavior rather
than silently relying on an imputer to conceal a training-serving skew.

The optional `evaluation_summary_path` is recorded only as operator-supplied
metadata. It does not cause the exporter to claim a registry version or query
MLflow.

## Runtime data boundary

Extend the API persistence boundary with a narrow protocol:

```python
class RecentHistoryRepository(Protocol):
    def get_recent_history(
        self,
        location_id: int,
        *,
        limit: int,
    ) -> list[HistoryRecord]: ...
```

`PostgresSensorHistoryRepository` implements it with a parameterized query for
one location, `ORDER BY observed_at DESC`, and `LIMIT limit`; it reverses the
result before returning it so callers always receive ascending timestamps. It
uses the same session lifecycle and `SQLAlchemyError -> DataStoreUnavailableError`
translation as the existing read methods.

The provider requests exactly `168` records. It requires all of the following
before inference:

- exactly 168 returned rows;
- timezone-aware timestamps on exact hour boundaries;
- strictly ascending timestamps one hour apart;
- finite, non-negative integer pedestrian counts.

The last observation is the live `data_cutoff_at` and `forecast_origin_at`.
This is deliberately stricter than the training pipeline's missing-value
markers: serving must not silently use median-imputed lag/rolling values after
a data gap.

Hourly continuity is checked on UTC instants, so a Melbourne daylight-saving
transition remains a one-hour sequence. After validation, every instant is
converted to the project's named `Australia/Melbourne` timezone before it is
given to `build_supervised_frame(...)`; therefore calendar hour, weekday, and
holiday features retain the same local-time semantics used by training even
when PostgreSQL returns UTC timestamps.

## Provider and feature construction

Introduce an `ArtifactBackedLightGBMForecastProvider` that consumes a validated
`FittedLightGBMModel`, the manifest, and a `RecentHistoryRepository`.

```python
class ArtifactBackedLightGBMForecastProvider:
    def predict(self, location_id: int, horizon: int) -> ForecastBatch: ...
```

For every call it:

1. obtains the validated 168-row live history window;
2. converts it to the existing observation DataFrame contract;
3. calls `build_supervised_frame(observations, horizons=range(1, horizon + 1),
   public_holidays=manifest.public_holidays)`;
4. selects only rows whose `forecast_origin_at` equals the live cutoff;
5. invokes the fitted model once for all direct horizon rows;
6. returns predictions sorted by `forecast_horizon`, with target timestamps
   taken from the generated direct rows.

The provider never inserts a predicted count into the history DataFrame. Thus
the existing direct multi-horizon rule remains intact: all lag and rolling
features are anchored to observed history at the cutoff, while calendar values
describe each known future target timestamp. Weather columns retain the current
explicitly-missing values and markers; this slice does not add weather data.
Before any history read, it rejects a boolean, non-integral, or out-of-range
(`1..24`) horizon. Forecast targets advance from the cutoff as UTC instants and
are then converted to `Australia/Melbourne`; local wall-clock `datetime +
timedelta` arithmetic is not used across daylight-saving changes. Before
feature construction, it verifies that each resulting Melbourne-local target
date is within the manifest calendar coverage; a manifest cannot silently
classify an uncovered future date as non-holiday.

Malformed model output is not a serving-input failure. A provider must reject
both short and overlong output, and values that cannot become floats, without
silently truncating output or fabricating a prediction row. Its dedicated
model-output error maps at the service boundary to `503 model_unavailable` with
message `"Forecast provider returned an invalid prediction batch."`. Finite-
value validation of an otherwise well-shaped batch remains at the existing
service boundary. None of these conditions may be relabeled as
`forecast_unavailable`.

`generated_at` is the UTC request-time timestamp. `model_name` is
`"lightgbm"`; `model_version` comes from the validated manifest and is never
invented. The API service retains its existing non-negative clipping as the
final serving boundary.

## Configuration and failure behavior

Default service construction observes two optional settings:

| Database URL | Artifact path | Sensor/history | Forecast behavior |
| --- | --- | --- | --- |
| absent or blank | absent or blank | empty defaults | existing `503 model_unavailable` |
| configured | absent or blank | PostgreSQL reads | existing `503 model_unavailable` |
| absent or blank | configured | empty defaults | `503 model_unavailable`; no artifact load is attempted |
| configured | configured and valid | PostgreSQL reads | real LightGBM forecasts |
| configured | configured but invalid | PostgreSQL reads | `503 model_unavailable`; no fabricated output |

An artifact is loaded only when both settings are non-blank. A missing,
corrupt, checksum-mismatched, unsupported, or feature-incompatible artifact
does not manufacture a provider; the existing `model_unavailable` contract is
retained. The sensor/history API remains usable in that state.

At request time:

| Condition | Response |
| --- | --- |
| no usable provider/artifact | existing `503 model_unavailable` |
| unknown sensor with a usable provider | existing `404 sensor_not_found` |
| PostgreSQL session/query failure | existing `503 data_store_unavailable` |
| fewer than 168 rows, a gap, invalid timestamp/count, or uncovered future holiday date | new `503 forecast_unavailable` with message `"Forecast cannot be generated from the available serving inputs."` |
| provider returns malformed, incomplete, or non-finite output | existing `503 model_unavailable` |

`ForecastService` catches `DataStoreUnavailableError` raised from provider
history reads and maps it to the existing data-store response. It catches a
new, provider-specific `ForecastInputUnavailableError` and maps it to the new
`forecast_unavailable` response. Route validation, unknown-sensor ordering,
OpenAPI paths, and unconfigured health behavior stay unchanged.

When no usable provider exists, `model_unavailable` deliberately takes
precedence over sensor lookup, matching the current `ForecastService` ordering.

## Testing and verification

Routine tests remain fully offline and PostgreSQL-free.

1. Artifact tests use temporary paths and a small deterministic LightGBM frame
   to verify export/load round trips, manifest fields, checksum/version mismatch,
   missing files, unsupported schema version, wrong model name, naive timestamps,
   invalid feature/config contract, malformed or uncovered holiday calendars,
   weather-incompatible input, refusal to overwrite an existing output
   directory, raw and Path-normalized URI-like paths, and parsing across a
   Melbourne daylight-saving offset change.
2. Provider tests inject an in-memory recent-history repository and a loaded
   temporary artifact. They prove all direct horizons are returned in order,
   cutoff and target timestamps are correct, no row after the cutoff affects
   features, predictions are non-negative at the API boundary, and targets
   crossing a Melbourne daylight-saving transition remain one-hour UTC
   instants.
3. Provider failure tests cover exactly 167 rows, an hourly gap, a naive or
   non-hour timestamp, a negative/non-integer count, an invalid horizon,
   calendar coverage expiry, repository failures, and a Melbourne daylight-
   saving transition supplied in UTC timestamps.
4. PostgreSQL adapter tests compile the recent-history statement and prove its
   location predicate, descending limit query, ascending returned records, and
   error translation.
5. App-factory/API tests cover every configuration row in the table without a
   real database or artifact path. Explicit injected `ApiServices` continues
   to bypass all environment-driven construction.
6. The full Ruff, format, pytest, and bounded Uvicorn `/health` smoke remain
   required. CI runs the same bounded default-configured health smoke after
   pytest. A manual PostgreSQL-and-artifact smoke is optional and only runs
   when the operator explicitly provides both local resources.

## Alternatives considered

### Serve both Ridge and LightGBM now

Rejected. It duplicates artifact schemas, configuration, runtime selection,
and test paths. LightGBM alone demonstrates the intended global serving
architecture; Ridge can be added later behind the same artifact boundary.

### Build a Streamlit Dashboard first

Rejected. The current Dashboard would show real sensor/history data but no
forecast result. A genuine provider gives the Dashboard a truthful data source
in the following slice.

### Use MLflow Model Registry as the first artifact source

Rejected. The current MLflow work records evaluation evidence and intentionally
has no registry or service dependency. An explicit local bundle is simpler to
validate, preserves offline tests, and can later be promoted into a registry
adapter without changing the forecast route.

### Forecast from a seasonal-naive fallback when the artifact is unavailable

Rejected. Returning a different model without an explicit operator decision
would obscure availability and model provenance. The route must return a clear
503 instead.

## Acceptance criteria

This slice is complete when:

- a local LightGBM artifact can be exported and checksum-validated;
- an explicitly configured valid artifact plus PostgreSQL history produces a
  real direct 1–24 hour API forecast;
- every response identifies the manifest model version and real data cutoff;
- no configured artifact or database preserves the existing safe defaults;
- insufficient/gapped history and unavailable storage never produce a fake
  forecast;
- routine tests use no network, PostgreSQL server, MLflow server, or committed
  model artifact;
- README distinguishes local artifact-backed serving from future registry,
  Dashboard, monitoring, and Docker work;
- Ruff, format checks, pytest, Uvicorn smoke, and GitHub Actions pass.
