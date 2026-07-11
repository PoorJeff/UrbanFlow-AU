# MLflow Tracking Design

Date: 2026-07-11

## Goal

Add the first MLflow experiment-tracking layer for UrbanFlow AU's local
modeling baselines.

The tracking layer should make the existing Ridge and LightGBM rolling-origin
evaluation outputs visible in MLflow without changing the leakage-safe
evaluation protocol. The first slice should record reproducible local
evaluation runs, metrics, parameters, tags, and report artifacts. It should not
yet introduce model registry workflows, Docker Compose services, API serving,
automatic retraining, or dashboard integration.

## Current project context

The repository already has:

- local ingestion, validation, PostgreSQL persistence, and Prefect orchestration
  slices;
- leakage-safe supervised feature builders;
- Seasonal Naive, Ridge, and LightGBM baseline modeling code;
- rolling-origin evaluation JSON CLIs for Ridge and LightGBM;
- Markdown report renderers for Ridge and LightGBM;
- checked-in synthetic evaluation examples;
- `mlruns/` ignored in `.gitignore`;
- full Ruff and pytest gates.

The requirements call for MLflow experiment records that include data version,
feature version, model parameters, backtest-window metrics, summary metrics,
training time, model files, and evaluation reports. The current local baseline
path can support evaluation-summary tracking immediately, but model-file
logging should be a separate follow-up because the existing CLIs serialize
evaluation summaries rather than persistent fitted-model artifacts.

## Reference check

Official MLflow documentation currently describes:

- runs as records of parameters, metrics, metadata, and artifacts;
- `mlflow.start_run`, `mlflow.log_param`, `mlflow.log_metric`, and
  `mlflow.log_artifact` as the core fluent tracking APIs;
- local `mlruns` as the default file-backed tracking store when no server is
  configured;
- LightGBM support through `mlflow.lightgbm.autolog` and
  `mlflow.lightgbm.log_model`, including feature-importance artifacts.

For this project, the first implementation should use explicit tracking calls
instead of autologging. Explicit calls keep the emitted metric names, tags, and
artifacts aligned with the project's existing JSON report contract. LightGBM
autologging and `log_model` remain useful follow-up tools once model-artifact
persistence is introduced.

## Selected approach

Implement a small, generic evaluation-tracking layer that consumes the already
generated evaluation JSON summary and optional Markdown report:

```powershell
python scripts/track_modeling_evaluation.py lightgbm reports/modeling/lightgbm_evaluation.json --report reports/modeling/lightgbm_evaluation.md
python scripts/track_modeling_evaluation.py ridge reports/modeling/ridge_evaluation.json --report reports/modeling/ridge_evaluation.md
```

This command should:

1. read an existing Ridge or LightGBM evaluation summary JSON;
2. validate the minimum rolling-origin summary fields;
3. set the MLflow tracking URI only when provided by CLI or environment;
4. set or create a stable experiment name, defaulting to
   `urbanflow-local-baselines`;
5. open one MLflow run for one evaluation summary;
6. log model/evaluation parameters and run tags;
7. log final-test metrics and validation-window metrics;
8. log the summary JSON and optional Markdown report as artifacts;
9. print a small JSON result containing `run_id`, `experiment_id`, and
   `tracking_uri`.

The command should not run model training. That keeps tracking deterministic
and lets users inspect or regenerate the JSON/Markdown artifacts before logging
them.

## MLflow run shape

### Experiment

- Default name: `urbanflow-local-baselines`.
- Override: `--experiment-name`.
- Tracking URI:
  - default: whatever MLflow would use locally, usually `mlruns/`;
  - override: `--tracking-uri` or `MLFLOW_TRACKING_URI`.

### Tags

Use short, searchable tags:

- `urbanflow.stage=local_baseline`;
- `urbanflow.model=ridge` or `urbanflow.model=lightgbm`;
- `urbanflow.summary_schema=rolling_origin_v1`;
- `urbanflow.source=supervised_csv`;
- optional `urbanflow.git_sha` if available cheaply through Git.

### Parameters

Log values that describe the run but are not metrics:

- `input_path`;
- `validation_window_count`;
- `row_count`;
- model hyperparameters present in the summary or supplied by CLI options;
- artifact paths for the local summary/report, if supplied.

Do not log the full supervised CSV as an artifact. Full raw and processed data
can be large and are intentionally kept out of Git and routine experiment
artifacts.

### Metrics

Log final-test metrics as stable scalar names:

- `final_test_row_count`;
- `final_test_mae`;
- `final_test_rmse`;
- `final_test_wape`;
- `final_test_seasonal_naive_mae`;
- `final_test_seasonal_naive_rmse`;
- `final_test_seasonal_naive_wape`;
- `final_test_relative_wape_improvement`.

Log validation-window metrics with `step` equal to the zero-based validation
window index:

- `validation_mae`;
- `validation_rmse`;
- `validation_wape`;
- `validation_seasonal_naive_wape`;
- `validation_relative_wape_improvement`.

The step-based representation keeps charting simple in the MLflow UI while the
artifact JSON remains the source of exact per-window names and dates.

### Artifacts

Log:

- the evaluation JSON summary under `evaluation/`;
- the Markdown report under `reports/` when supplied.

Later model-artifact slices can log:

- a final fitted Ridge or LightGBM model;
- feature importance JSON/PNG;
- model signature/input example;
- model card snippets.

## Error behavior

The tracking CLI should return exit code `2` for expected user errors:

- summary JSON path does not exist;
- invalid JSON;
- summary JSON is not an object;
- unsupported model name;
- required rolling-origin summary fields are missing;
- report path is supplied but does not exist;
- MLflow logging raises a known MLflow exception.

Unexpected exceptions should continue to fail normally during tests, rather
than being swallowed silently.

## Alternatives considered

### 1. Add MLflow logging directly into Ridge and LightGBM evaluation CLIs

Rejected for the first slice. The evaluation CLIs are currently pure local
commands that print JSON to stdout. Adding side effects to them would make it
easier to accidentally log every test or exploratory run. A separate tracking
CLI keeps evaluation and experiment logging explicit.

### 2. Use `mlflow.lightgbm.autolog` immediately

Rejected for the first slice. Autologging can log useful LightGBM model and
feature-importance artifacts, but the project first needs consistent run names,
metric names, tags, and report artifacts across Ridge and LightGBM.

### 3. Add a local MLflow tracking server and Docker Compose service now

Rejected for the next implementation slice. Local file-backed `mlruns/` is
enough for individual development and unit testing. The tracking server belongs
with the later Docker Compose/API/dashboard integration stage.

### 4. Commit generated MLflow runs or model files to Git

Rejected. `mlruns/` and model artifacts remain local generated outputs. The
repository should only track small deterministic examples, documentation, and
source code.

## Testing strategy

Use TDD in small slices:

1. Unit-test JSON summary loading and validation.
2. Unit-test metric flattening from Ridge and LightGBM summaries.
3. Unit-test tracking with a fake MLflow adapter so most tests do not write
   local runs.
4. CLI-test expected user errors and success JSON output.
5. Add one optional local-file smoke test using a temporary tracking URI if it
   stays fast and deterministic.

Tests should not require a running MLflow server, PostgreSQL, network access,
or a real supervised dataset.

## Acceptance criteria

The first MLflow tracking slice is complete when:

- MLflow is added as a runtime dependency;
- a tracking module can log Ridge and LightGBM evaluation summaries;
- metrics, params, tags, JSON artifacts, and Markdown report artifacts are
  captured in a local MLflow run;
- the CLI returns machine-readable success output;
- expected invalid inputs return exit code `2`;
- `README.md` explains how to evaluate, render, and track a baseline run;
- `mlruns/` remains ignored and no generated run artifacts are committed;
- full Ruff and pytest suites pass.

## Follow-up slices

After summary/report tracking is stable:

1. Log final fitted model artifacts for Ridge and LightGBM.
2. Add LightGBM feature-importance artifacts.
3. Add a local MLflow UI command or Docker Compose service.
4. Add model-card and data-card links to tracked artifacts.
5. Feed tracked final-test metrics into the future API/dashboard model metadata
   endpoints.
