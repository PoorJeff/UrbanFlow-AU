# Ridge Evaluation CLI Design

Date: 2026-07-02

## Goal

Add a local command-line entrypoint that evaluates the leakage-safe Ridge
baseline from a supervised feature-row CSV and prints a deterministic JSON
summary.

This slice turns the current Ridge library code into something a project user
can run directly. It should stay local, file-based, and small: no PostgreSQL
reader, no MLflow, no model artifact persistence, no charts, and no dashboard
code.

## Current project context

The repository already has:

- `urbanflow.features.build_supervised_frame`, which creates supervised
  multi-horizon rows from hourly observations;
- `urbanflow.modeling.build_rolling_origin_splits`, which derives validation
  and final-test windows from complete calendar months;
- `urbanflow.modeling.evaluate_rolling_origin_ridge`, which fits Ridge on rows
  before each window and evaluates predictions inside each window;
- CLI conventions where `src/urbanflow/<domain>/cli.py` owns `argparse`, JSON
  summaries, and exit codes, while `scripts/*.py` files are thin wrappers.

The repository does not yet have a runnable modeling script. A user can import
the Ridge helpers from Python, but cannot yet run Ridge evaluation from the
terminal against a local supervised CSV.

## Selected approach

Create a small modeling CLI:

- `src/urbanflow/modeling/cli.py` owns parsing, CSV loading, Ridge evaluation,
  JSON summary formatting, and CLI exit codes.
- `scripts/evaluate_ridge_baseline.py` is a thin wrapper that calls
  `urbanflow.modeling.cli.main`.
- `README.md` gets a short local command example.

The CLI will accept a CSV that already follows the supervised frame contract.
It will not build supervised rows from raw hourly observations in this slice;
that keeps the command focused on model evaluation and avoids mixing feature
generation with model evaluation semantics.

## Alternatives considered

### 1. Ridge evaluation CLI from supervised CSV

This is the selected approach. It creates the smallest useful runnable surface
on top of the completed Ridge baseline while preserving the existing
DataFrame-first modeling contract.

### 2. End-to-end CLI from raw hourly observations

This would be convenient, but it would combine raw observation parsing,
feature-row construction, split derivation, model fitting, and reporting in one
change. That is a better later command after the modeling CLI is stable.

### 3. Database-backed Ridge evaluation CLI

This would look more production-like, but it would couple the modeling slice to
PostgreSQL availability. A database reader can be added later as a separate
input source that produces the same supervised DataFrame.

## CLI contract

The primary command should be:

```powershell
python scripts/evaluate_ridge_baseline.py data/modeling/supervised_rows.csv --validation-months 3
```

Arguments:

- positional `supervised_csv`: path to a local CSV file containing supervised
  rows;
- optional `--validation-months`: positive integer, default `3`, passed to
  `build_rolling_origin_splits`;
- optional `--alpha`: positive float, default `1.0`, passed to
  `RidgeModelConfig`.

The command prints one JSON object to stdout on success and returns exit code
`0`.

## CSV input behavior

`modeling.cli` should read the CSV with pandas and parse these timestamp
columns when present:

- `forecast_origin_at`;
- `target_observed_at`.

The loaded DataFrame must satisfy the existing Ridge supervised-frame contract:

- it includes the default Ridge feature columns;
- it includes `target` and `target_missing`;
- `target_observed_at` is timezone-aware after CSV parsing;
- at least two complete calendar months are available for split generation.

The CLI does not impute targets, repair timestamps, or infer timezones. If a
CSV loses timezone information, the command fails with a clear input error
instead of silently assuming a timezone.

## JSON summary shape

The success payload should be stable and compact:

```json
{
  "input_path": "data/modeling/supervised_rows.csv",
  "row_count": 1464,
  "validation_window_count": 1,
  "validation_windows": [
    {
      "name": "validation_2025-01",
      "start": "2025-01-01T00:00:00+11:00",
      "end": "2025-02-01T00:00:00+11:00",
      "train_end": "2025-01-01T00:00:00+11:00",
      "training_row_count": 744,
      "overall": {
        "row_count": 744,
        "mae": 1.23,
        "rmse": 1.75,
        "wape": 0.08
      },
      "horizon_metrics": [
        {
          "forecast_horizon": 1,
          "row_count": 744,
          "mae": 1.23,
          "rmse": 1.75,
          "wape": 0.08
        }
      ]
    }
  ],
  "final_test": {
    "name": "final_test_2025-02",
    "start": "2025-02-01T00:00:00+11:00",
    "end": "2025-03-01T00:00:00+11:00",
    "train_end": "2025-02-01T00:00:00+11:00",
    "training_row_count": 744,
    "overall": {
      "row_count": 672,
      "mae": 1.2,
      "rmse": 1.7,
      "wape": 0.07
    },
    "horizon_metrics": [
      {
        "forecast_horizon": 1,
        "row_count": 672,
        "mae": 1.2,
        "rmse": 1.7,
        "wape": 0.07
      }
    ]
  }
}
```

The exact metric values depend on the input data. `None` metric values should
serialize as JSON `null`. Timestamps should serialize with `Timestamp.isoformat`
so timezone offsets remain visible.

## Error handling and exit codes

The command should avoid tracebacks for expected user errors.

Return code `2` for invalid command input or invalid data, including:

- missing CSV file;
- unreadable CSV;
- non-positive `--validation-months`;
- non-positive `--alpha`;
- missing required columns;
- timezone-naive `target_observed_at`;
- not enough complete months for rolling-origin splits;
- no training or evaluation rows for a derived window.

On these failures, print a concise message to stderr and print no success JSON
to stdout.

No separate code `1` is needed in this slice because the command has no network,
database, or external service runtime path. A later database-backed variant can
reserve code `1` for operational failures.

## Testing strategy

Unit tests should follow the existing CLI style:

- call `urbanflow.modeling.cli.main` directly for success and error paths;
- use a temporary CSV generated from a deterministic synthetic supervised
  DataFrame with two complete calendar months and one forecast horizon;
- assert success returns `0` and stdout contains `validation_windows`,
  `final_test`, and finite metric fields;
- assert missing input returns `2`, writes a concise stderr message, and emits
  no success JSON;
- assert invalid option values return `2`;
- run `python scripts/evaluate_ridge_baseline.py --help` with subprocess and
  assert the help text names Ridge evaluation.

The implementation plan must use TDD for the CLI behavior: write failing tests,
verify the expected red state, implement the minimal CLI, then run focused and
full quality gates.

## README update

Add a short example below the Ridge baseline README section:

```powershell
python scripts/evaluate_ridge_baseline.py data/modeling/supervised_rows.csv --validation-months 3
```

The README should clarify that the input is an already-built supervised feature
CSV, not raw City of Melbourne hourly-count data.

## Out of scope

This slice intentionally does not add:

- raw hourly observation to supervised feature generation from the CLI;
- prediction CSV output;
- charts, screenshots, or dashboard views;
- MLflow experiment logging;
- LightGBM;
- PostgreSQL training reads;
- model artifact persistence;
- hyperparameter search;
- comparison tables against Seasonal Naive.

Those pieces become safer once the basic Ridge evaluation command is stable and
tested.

## Success criteria

The implementation is successful when:

- a user can run a local Ridge evaluation command against a supervised CSV;
- the command emits a deterministic JSON summary with validation and final-test
  metrics;
- expected user input errors return code `2` without a traceback;
- CLI tests cover success, invalid input, and script help behavior;
- README documents the command and its input contract;
- the full repository quality gate passes before merge and again on `main`.
