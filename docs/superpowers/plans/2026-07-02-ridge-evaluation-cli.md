# Ridge Evaluation CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CLI that evaluates the Ridge baseline from a supervised feature-row CSV and prints a deterministic JSON summary.

**Architecture:** Add `urbanflow.modeling.cli` as the command boundary for argument parsing, CSV loading, rolling-origin Ridge evaluation, JSON serialization, and expected user-error exit codes. Add `scripts/evaluate_ridge_baseline.py` as a thin wrapper, then document the command in `README.md`.

**Tech Stack:** Python 3.11+, argparse, pandas, existing `urbanflow.modeling` Ridge/split/evaluation helpers, pytest, Ruff.

---

## Source spec

Implement:

`docs/superpowers/specs/2026-07-02-ridge-evaluation-cli-design.md`

## Worktree and execution note

Create an isolated worktree before executing implementation tasks:

```powershell
git worktree add '.worktrees/ridge-evaluation-cli' -b codex/ridge-evaluation-cli
cd '.worktrees/ridge-evaluation-cli'
$env:PYTHONPATH='src'
```

Use the existing project virtual environment:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest
```

When finishing, merge the local `codex/ridge-evaluation-cli` branch into
`main` and push only `main`.

## File structure

- Create `src/urbanflow/modeling/cli.py`
  - Owns `argparse`, supervised CSV loading, positive option checks, Ridge evaluation orchestration, JSON summary serialization, and expected user-error exit code `2`.
- Create `scripts/evaluate_ridge_baseline.py`
  - Thin executable wrapper that imports `urbanflow.modeling.cli.main`.
- Create `tests/unit/modeling/test_cli.py`
  - Tests success JSON, missing input, invalid options, and script help text.
- Modify `README.md`
  - Adds a short Ridge evaluation command example and clarifies that the input is an already-built supervised feature CSV.

## Task 1: CLI behavior tests

**Files:**
- Create: `tests/unit/modeling/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/unit/modeling/test_cli.py`:

```python
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from urbanflow.modeling.cli import main


def supervised_rows() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-12-01 00:00",
        "2025-02-28 23:00",
        freq="h",
        tz="Australia/Melbourne",
    )
    values = [
        100.0 + float(index % 24) + float((index // 24) % 7)
        for index in range(len(timestamps))
    ]
    return pd.DataFrame(
        {
            "location_id": [101] * len(timestamps),
            "forecast_origin_at": timestamps - pd.Timedelta(hours=1),
            "forecast_horizon": [1] * len(timestamps),
            "target_observed_at": timestamps,
            "target": values,
            "target_missing": [False] * len(timestamps),
            "pedestrian_count": [value - 1.0 for value in values],
            "pedestrian_count_missing": [False] * len(timestamps),
            "lag_1": [value - 1.0 for value in values],
            "lag_24": [value - 2.0 for value in values],
            "lag_168": [value - 3.0 for value in values],
            "rolling_24_mean": [value - 1.5 for value in values],
            "rolling_24_std": [2.0] * len(timestamps),
            "rolling_168_mean": [value - 2.5 for value in values],
            "rolling_168_std": [4.0] * len(timestamps),
            "hour": timestamps.hour,
            "weekday": timestamps.weekday,
            "month": timestamps.month,
            "is_weekend": [timestamp.weekday() >= 5 for timestamp in timestamps],
            "is_public_holiday": [False] * len(timestamps),
            "hour_sin": [math.sin((timestamp.hour / 24.0) * math.tau) for timestamp in timestamps],
            "hour_cos": [math.cos((timestamp.hour / 24.0) * math.tau) for timestamp in timestamps],
            "weekday_sin": [
                math.sin((timestamp.weekday() / 7.0) * math.tau)
                for timestamp in timestamps
            ],
            "weekday_cos": [
                math.cos((timestamp.weekday() / 7.0) * math.tau)
                for timestamp in timestamps
            ],
            "temperature": [20.0] * len(timestamps),
            "temperature_missing": [False] * len(timestamps),
            "rainfall": [0.0] * len(timestamps),
            "rainfall_missing": [False] * len(timestamps),
            "wind_speed": [12.0] * len(timestamps),
            "wind_speed_missing": [False] * len(timestamps),
        }
    )


def write_supervised_csv(tmp_path: Path) -> Path:
    path = tmp_path / "supervised_rows.csv"
    supervised_rows().to_csv(path, index=False)
    return path


def assert_finite_metric(value: object) -> None:
    assert isinstance(value, float)
    assert math.isfinite(value)


def test_ridge_evaluation_cli_returns_json_summary(tmp_path, capsys) -> None:
    path = write_supervised_csv(tmp_path)

    exit_code = main([str(path), "--validation-months", "1", "--alpha", "0.5"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["input_path"] == str(path)
    assert payload["row_count"] == len(supervised_rows())
    assert payload["validation_window_count"] == 1
    assert payload["validation_windows"][0]["name"] == "validation_2025-01"
    assert payload["validation_windows"][0]["training_row_count"] == 744
    assert payload["validation_windows"][0]["overall"]["row_count"] == 744
    assert payload["validation_windows"][0]["horizon_metrics"][0]["forecast_horizon"] == 1
    assert payload["final_test"]["name"] == "final_test_2025-02"
    assert payload["final_test"]["training_row_count"] == 1488
    assert payload["final_test"]["overall"]["row_count"] == 672
    assert payload["final_test"]["horizon_metrics"][0]["row_count"] == 672
    assert_finite_metric(payload["final_test"]["overall"]["mae"])
    assert_finite_metric(payload["final_test"]["overall"]["rmse"])
    assert_finite_metric(payload["final_test"]["overall"]["wape"])


def test_ridge_evaluation_cli_returns_two_for_missing_input(tmp_path, capsys) -> None:
    exit_code = main([str(tmp_path / "missing.csv")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "CSV file does not exist" in captured.err


def test_ridge_evaluation_cli_returns_two_for_invalid_options(tmp_path, capsys) -> None:
    path = write_supervised_csv(tmp_path)

    exit_code = main([str(path), "--validation-months", "0", "--alpha", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "validation-months must be greater than zero" in captured.err


def test_evaluate_ridge_baseline_script_help() -> None:
    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, repository_root / "scripts" / "evaluate_ridge_baseline.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Evaluate a local Ridge baseline" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_cli.py -v
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'urbanflow.modeling.cli'`.

- [ ] **Step 3: Commit nothing after the red test**

Run:

```powershell
git status --short
```

Expected: `tests/unit/modeling/test_cli.py` is untracked and no production CLI file exists yet.

Do not commit the red test alone; keep it staged later with the implementation that turns it green.

## Task 2: Modeling CLI implementation

**Files:**
- Create: `src/urbanflow/modeling/cli.py`
- Modify: `tests/unit/modeling/test_cli.py`

- [ ] **Step 1: Implement the minimal CLI**

Create `src/urbanflow/modeling/cli.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from urbanflow.modeling.evaluation import (
    ModelWindowEvaluation,
    RollingOriginRidgeEvaluation,
    evaluate_rolling_origin_ridge,
)
from urbanflow.modeling.feature_matrix import ModelTrainingError
from urbanflow.modeling.metrics import RegressionMetrics
from urbanflow.modeling.ridge import RidgeModelConfig
from urbanflow.modeling.splits import SplitConfigError, build_rolling_origin_splits

TIMESTAMP_COLUMNS = ("forecast_origin_at", "target_observed_at")


class RidgeEvaluationCliError(ValueError):
    """Raised when local Ridge evaluation CLI input is invalid."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a local Ridge baseline from supervised feature rows."
    )
    parser.add_argument(
        "supervised_csv",
        type=Path,
        help="CSV containing already-built supervised feature rows.",
    )
    parser.add_argument(
        "--validation-months",
        type=int,
        default=3,
        help="Positive number of validation months before the final test month.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Positive Ridge regularization strength.",
    )
    return parser


def _positive_integer(value: int, *, name: str) -> int:
    if value <= 0:
        raise RidgeEvaluationCliError(f"{name} must be greater than zero")
    return value


def _positive_float(value: float, *, name: str) -> float:
    if value <= 0:
        raise RidgeEvaluationCliError(f"{name} must be greater than zero")
    return value


def _read_supervised_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RidgeEvaluationCliError(f"CSV file does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise RidgeEvaluationCliError(f"could not read supervised CSV: {path}") from exc

    for column in TIMESTAMP_COLUMNS:
        if column in frame.columns:
            try:
                frame[column] = pd.to_datetime(frame[column])
            except (TypeError, ValueError) as exc:
                raise RidgeEvaluationCliError(f"could not parse timestamp column: {column}") from exc
    return frame


def _json_scalar(value: object) -> object:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _metrics_summary(metrics: RegressionMetrics) -> dict[str, object]:
    return {
        "row_count": metrics.row_count,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "wape": metrics.wape,
    }


def _horizon_metric_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        records.append({key: _json_scalar(value) for key, value in record.items()})
    return records


def _timestamp_text(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).isoformat()


def _window_summary(evaluation: ModelWindowEvaluation) -> dict[str, object]:
    window = evaluation.window
    return {
        "name": window.name,
        "start": _timestamp_text(window.start),
        "end": _timestamp_text(window.end),
        "train_end": _timestamp_text(window.train_end),
        "training_row_count": evaluation.model.training_row_count,
        "overall": _metrics_summary(evaluation.overall_metrics),
        "horizon_metrics": _horizon_metric_records(evaluation.horizon_metrics),
    }


def evaluation_summary(
    evaluation: RollingOriginRidgeEvaluation,
    *,
    input_path: Path,
    row_count: int,
) -> dict[str, Any]:
    validation_windows = [
        _window_summary(window_evaluation) for window_evaluation in evaluation.validation_windows
    ]
    return {
        "input_path": str(input_path),
        "row_count": row_count,
        "validation_window_count": len(validation_windows),
        "validation_windows": validation_windows,
        "final_test": _window_summary(evaluation.final_test),
    }


def run_ridge_evaluation(
    supervised_csv: Path,
    *,
    validation_months: int,
    alpha: float,
) -> dict[str, Any]:
    supervised_frame = _read_supervised_csv(supervised_csv)
    splits = build_rolling_origin_splits(
        supervised_frame,
        validation_months=validation_months,
    )
    evaluation = evaluate_rolling_origin_ridge(
        supervised_frame,
        splits,
        model_config=RidgeModelConfig(alpha=alpha),
    )
    return evaluation_summary(evaluation, input_path=supervised_csv, row_count=len(supervised_frame))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validation_months = _positive_integer(
            args.validation_months,
            name="validation-months",
        )
        alpha = _positive_float(args.alpha, name="alpha")
        summary = run_ridge_evaluation(
            args.supervised_csv,
            validation_months=validation_months,
            alpha=alpha,
        )
    except (ModelTrainingError, RidgeEvaluationCliError, SplitConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, sort_keys=True))
    return 0
```

- [ ] **Step 2: Run focused CLI tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_cli.py -v
```

Expected: three tests pass and `test_evaluate_ridge_baseline_script_help` fails because `scripts/evaluate_ridge_baseline.py` does not exist.

## Task 3: Script wrapper and README documentation

**Files:**
- Create: `scripts/evaluate_ridge_baseline.py`
- Modify: `README.md`
- Modify: `tests/unit/modeling/test_cli.py`

- [ ] **Step 1: Add the script wrapper**

Create `scripts/evaluate_ridge_baseline.py`:

```python
from urbanflow.modeling.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run focused CLI tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_cli.py -v
```

Expected: all four CLI tests pass.

- [ ] **Step 3: Add the README command example**

In `README.md`, under `## Train a local Ridge baseline`, append this paragraph and command block after the existing Ridge baseline explanation:

````markdown
To evaluate Ridge from an already-built supervised feature CSV, run:

```powershell
python scripts/evaluate_ridge_baseline.py data/modeling/supervised_rows.csv --validation-months 3
```

The command expects supervised feature rows, not raw City of Melbourne
hourly-count data. It prints a JSON summary with rolling-origin validation and
final-test metrics.
````

- [ ] **Step 4: Run the CLI help command manually**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' scripts/evaluate_ridge_baseline.py --help
```

Expected: stdout contains `Evaluate a local Ridge baseline from supervised feature rows.`

- [ ] **Step 5: Run targeted modeling checks**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_cli.py tests/unit/modeling/test_evaluation.py tests/unit/modeling/test_ridge.py -v
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check src/urbanflow/modeling tests/unit/modeling scripts/evaluate_ridge_baseline.py --no-cache
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check src/urbanflow/modeling tests/unit/modeling scripts/evaluate_ridge_baseline.py
```

Expected: pytest passes, Ruff check passes, and Ruff format check passes.

- [ ] **Step 6: Commit the CLI implementation**

Run:

```powershell
git add src/urbanflow/modeling/cli.py scripts/evaluate_ridge_baseline.py tests/unit/modeling/test_cli.py README.md
git commit -m "feat: add ridge evaluation cli"
```

Expected: one commit containing the tested modeling CLI, script wrapper, CLI tests, and README example.

## Task 4: Full verification, merge, push, and cleanup

**Files:**
- Verify: repository quality gate
- Merge: `codex/ridge-evaluation-cli` into `main`
- Push: `main`

- [ ] **Step 1: Run the full quality gate on the implementation branch**

Run:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check . --no-cache
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check .
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest
```

Expected: Ruff passes, format check passes, and all tests pass.

- [ ] **Step 2: Self-review implementation scope**

Run:

```powershell
git status --short --branch
git diff --stat main...HEAD
git log --oneline -4
rg -n "MLflow|LightGBM|PostgreSQL|artifact|prediction CSV|hyperparameter" src/urbanflow/modeling scripts README.md tests/unit/modeling
```

Expected:

- working tree is clean on `codex/ridge-evaluation-cli`;
- diff contains only `src/urbanflow/modeling/cli.py`, `scripts/evaluate_ridge_baseline.py`, `tests/unit/modeling/test_cli.py`, and `README.md`;
- search hits are limited to README prose or no hits, with no new MLflow, LightGBM, PostgreSQL reader, artifact persistence, prediction CSV writer, or hyperparameter search code.

- [ ] **Step 3: Verify remote main has not moved**

Run from `D:\Github项目\UrbanFlow-AU`:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 fetch origin main
git rev-list --left-right --count main...origin/main
```

Expected: `0 0`.

- [ ] **Step 4: Merge to main only**

Run from `D:\Github项目\UrbanFlow-AU`:

```powershell
git merge --ff-only codex/ridge-evaluation-cli
git status --short --branch
```

Expected: `main` fast-forwards and is ahead of `origin/main` by one commit.

- [ ] **Step 5: Re-run the full quality gate on main**

Run from `D:\Github项目\UrbanFlow-AU`:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check . --no-cache
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check .
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest
```

Expected: Ruff passes, format check passes, and all tests pass on `main`.

- [ ] **Step 6: Push main**

Run from `D:\Github项目\UrbanFlow-AU`:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 push origin main
```

Expected: only `main` is pushed to GitHub.

- [ ] **Step 7: Remove the local worktree and local codex branch**

Before removal, verify the resolved path stays under `D:\Github项目\UrbanFlow-AU\.worktrees`:

```powershell
$target = (Resolve-Path '.worktrees\ridge-evaluation-cli').Path
$root = (Resolve-Path '.worktrees').Path
if (-not $target.StartsWith($root)) { throw "Refusing to remove worktree outside .worktrees: $target" }
git worktree remove $target
git worktree prune
git branch -d codex/ridge-evaluation-cli
```

Expected: local implementation worktree and local codex branch are removed after the successful merge and push.

## Self-review checklist

Before executing Task 4 merge/push:

- Spec coverage:
  - supervised CSV input: Task 2.
  - timestamp parsing for `forecast_origin_at` and `target_observed_at`: Task 2.
  - `--validation-months`: Task 2 tests and implementation.
  - `--alpha`: Task 2 tests and implementation.
  - JSON summary with validation windows and final test: Task 1 tests and Task 2 implementation.
  - user-error exit code `2`: Task 1 tests and Task 2 implementation.
  - thin script wrapper: Task 3.
  - README command example: Task 3.
- Type consistency:
  - `RidgeEvaluationCliError` is local to `urbanflow.modeling.cli`.
  - `evaluation_summary` accepts `RollingOriginRidgeEvaluation`.
  - `_window_summary` accepts `ModelWindowEvaluation`.
  - metric summaries use existing `RegressionMetrics`.
  - Ridge config uses existing `RidgeModelConfig(alpha=alpha)`.
- Scope guard:
  - no database reads;
  - no MLflow logging;
  - no LightGBM dependency;
  - no model artifact persistence;
  - no prediction CSV writer;
  - no chart or dashboard code.
