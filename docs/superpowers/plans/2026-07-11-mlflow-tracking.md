# MLflow Tracking Implementation Plan

> **For agentic workers:** implement task-by-task with small commits. Do not use
> subagents unless the user explicitly asks for parallel agent work.

**Goal:** Add explicit MLflow tracking for local Ridge and LightGBM evaluation
summaries and Markdown reports.

**Architecture:** Keep evaluation and tracking separate. Existing evaluation
commands write JSON; existing report commands render Markdown; a new tracking
command consumes those artifacts and records one MLflow run.

**Design source:** `docs/superpowers/specs/2026-07-11-mlflow-tracking-design.md`

## Task 1: Add MLflow dependency and tracking adapter seam

**Files:**

- Modify: `pyproject.toml`
- Add: `src/urbanflow/modeling/mlflow_tracking.py`
- Add: `tests/unit/modeling/test_mlflow_tracking.py`

Steps:

- [ ] Add `mlflow>=3,<4` as a runtime dependency.
- [ ] Add a small adapter/protocol boundary so unit tests can use a fake logger.
- [ ] Add dataclasses:
  - `MLflowTrackingConfig`;
  - `MLflowRunResult`.
- [ ] Add summary-loading and validation helpers.
- [ ] Add metric-flattening helpers for:
  - final-test metrics;
  - Seasonal Naive final-test metrics;
  - relative WAPE improvement;
  - validation-window metrics with steps.
- [ ] Unit-test all helpers without creating real `mlruns/` output.

Verification:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_mlflow_tracking.py -q
```

## Task 2: Implement explicit MLflow run logging

**Files:**

- Modify: `src/urbanflow/modeling/mlflow_tracking.py`
- Modify: `tests/unit/modeling/test_mlflow_tracking.py`

Steps:

- [ ] Implement `track_evaluation_summary(...)`.
- [ ] Set the tracking URI only when explicitly provided.
- [ ] Set or create the configured experiment.
- [ ] Start one run per evaluation summary.
- [ ] Log params, tags, metrics, and artifacts through the adapter seam.
- [ ] Ensure missing optional Markdown report skips only that artifact.
- [ ] Convert expected MLflow exceptions into project-level tracking errors.

Verification:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_mlflow_tracking.py -q
```

## Task 3: Add tracking CLI and script wrapper

**Files:**

- Add: `src/urbanflow/modeling/mlflow_tracking_cli.py`
- Add: `scripts/track_modeling_evaluation.py`
- Add/modify: CLI tests under `tests/unit/modeling/`

Command shape:

```powershell
python scripts/track_modeling_evaluation.py lightgbm reports/modeling/lightgbm_evaluation.json --report reports/modeling/lightgbm_evaluation.md
python scripts/track_modeling_evaluation.py ridge reports/modeling/ridge_evaluation.json --report reports/modeling/ridge_evaluation.md
```

Required options:

- positional `model_name`: `ridge` or `lightgbm`;
- positional `summary_json`;
- optional `--report`;
- optional `--experiment-name`;
- optional `--tracking-uri`;
- optional repeated `--tag key=value` if simple to add without broad parsing
  complexity.

Expected CLI behavior:

- success prints JSON with `run_id`, `experiment_id`, and `tracking_uri`;
- expected user errors return exit code `2`;
- help text explains that the command logs existing evaluation artifacts and
  does not run training.

Verification:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling -q
```

## Task 4: Add a local-file smoke test if fast

**Files:**

- Modify: `tests/unit/modeling/test_mlflow_tracking.py` or add an integration
  style test under `tests/unit/modeling/`

Steps:

- [ ] Use a temporary directory as `file://` tracking URI.
- [ ] Log the synthetic LightGBM example summary and report.
- [ ] Assert that a run ID is returned.
- [ ] Assert that no run output lands outside the temporary directory.
- [ ] Keep the test small enough for routine CI.

If this proves flaky or slow, keep adapter-based unit coverage and document the
manual smoke command instead.

## Task 5: README usage and final gate

**Files:**

- Modify: `README.md`

Add a short section after the Ridge/LightGBM report commands:

```powershell
python scripts/track_modeling_evaluation.py lightgbm reports/modeling/lightgbm_evaluation.json --report reports/modeling/lightgbm_evaluation.md
mlflow server --port 5000
```

Clarify:

- `mlruns/` is local generated output and ignored by Git;
- tracking logs local evaluation evidence, not production performance claims;
- model registry and Docker Compose MLflow service are future slices.

Final verification:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check .
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check .
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest -q
git status --short
git diff --check
```

## Self-review checklist

- [ ] The tracking command does not retrain models.
- [ ] The supervised CSV itself is not logged as an artifact.
- [ ] Expected user errors return code `2`.
- [ ] Ridge and LightGBM both use the same tracking path.
- [ ] Metric names are stable and documented.
- [ ] Validation-window metrics use MLflow steps.
- [ ] JSON and Markdown reports are logged as artifacts.
- [ ] No generated `mlruns/`, model files, data snapshots, or secrets are
      committed.
