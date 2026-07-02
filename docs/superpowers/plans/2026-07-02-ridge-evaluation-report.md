# Ridge Evaluation Markdown Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Markdown report renderer for Ridge evaluation JSON summaries.

**Architecture:** Add a pure `urbanflow.modeling.reports` renderer that validates Ridge evaluation summary dictionaries and formats deterministic Markdown. Add `urbanflow.modeling.report_cli` plus a thin script wrapper to read JSON summaries and write Markdown reports without running model evaluation.

**Tech Stack:** Python 3.11+, standard-library `argparse`/`json`/`pathlib`, existing Ridge evaluation JSON summary contract, pytest, Ruff.

---

## Source spec

Implement:

`docs/superpowers/specs/2026-07-02-ridge-evaluation-report-design.md`

## Worktree and execution note

Create an isolated worktree before executing implementation tasks:

```powershell
git worktree add '.worktrees/ridge-evaluation-report' -b codex/ridge-evaluation-report
cd '.worktrees/ridge-evaluation-report'
$env:PYTHONPATH='src'
```

Use the existing project virtual environment:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest
```

When finishing, merge the local `codex/ridge-evaluation-report` branch into
`main` and push only `main`.

## File structure

- Create `src/urbanflow/modeling/reports.py`
  - Owns Ridge summary validation, Markdown scalar formatting, Markdown table formatting, and `render_ridge_evaluation_report`.
- Create `src/urbanflow/modeling/report_cli.py`
  - Owns `argparse`, JSON reading, output path resolution, overwrite protection, Markdown file writes, and expected user-error exit code `2`.
- Create `scripts/render_ridge_evaluation_report.py`
  - Thin executable wrapper that imports `urbanflow.modeling.report_cli.main`.
- Create `tests/unit/modeling/test_modeling_reports.py`
  - Uses a unique basename to avoid collision with `tests/unit/validation/test_reports.py`.
  - Tests renderer behavior, CLI behavior, and script help.
- Modify `README.md`
  - Adds a short JSON-to-Markdown Ridge report workflow below the existing Ridge CLI example.

## Task 1: Markdown renderer contract

**Files:**
- Create: `tests/unit/modeling/test_modeling_reports.py`
- Create: `src/urbanflow/modeling/reports.py`

- [ ] **Step 1: Write failing renderer tests**

Create `tests/unit/modeling/test_modeling_reports.py`:

```python
from __future__ import annotations

from copy import deepcopy

import pytest

from urbanflow.modeling.reports import RidgeReportError, render_ridge_evaluation_report


def ridge_summary() -> dict[str, object]:
    return {
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
                    "mae": 1.23456,
                    "rmse": 1.75432,
                    "wape": 0.08123,
                },
                "horizon_metrics": [
                    {
                        "forecast_horizon": 1,
                        "row_count": 744,
                        "mae": 1.23456,
                        "rmse": 1.75432,
                        "wape": 0.08123,
                    }
                ],
            }
        ],
        "final_test": {
            "name": "final_test_2025-02",
            "start": "2025-02-01T00:00:00+11:00",
            "end": "2025-03-01T00:00:00+11:00",
            "train_end": "2025-02-01T00:00:00+11:00",
            "training_row_count": 1488,
            "overall": {
                "row_count": 672,
                "mae": 1.2,
                "rmse": 1.7,
                "wape": 0.07,
            },
            "horizon_metrics": [
                {
                    "forecast_horizon": 1,
                    "row_count": 672,
                    "mae": 1.2,
                    "rmse": 1.7,
                    "wape": 0.07,
                }
            ],
        },
    }


def test_render_ridge_evaluation_report_includes_core_sections() -> None:
    markdown = render_ridge_evaluation_report(ridge_summary())

    assert markdown.startswith("# Ridge Evaluation Report\n")
    assert "Source: `data/modeling/supervised_rows.csv`" in markdown
    assert "Rows evaluated: 1464" in markdown
    assert "Validation windows: 1" in markdown
    assert "## Final test" in markdown
    assert "Window: `final_test_2025-02`" in markdown
    assert "| Row count | 672 |" in markdown
    assert "| MAE | 1.2000 |" in markdown
    assert "## Validation windows" in markdown
    assert (
        "| validation_2025-01 | 2025-01-01T00:00:00+11:00 to "
        "2025-02-01T00:00:00+11:00 | 744 | 744 | 1.2346 | 1.7543 | 0.0812 |"
        in markdown
    )
    assert "## Final test by horizon" in markdown
    assert "| 1 | 672 | 1.2000 | 1.7000 | 0.0700 |" in markdown
    assert markdown.endswith("\n")


def test_render_ridge_evaluation_report_formats_missing_metrics_as_na() -> None:
    summary = deepcopy(ridge_summary())
    final_test = summary["final_test"]
    assert isinstance(final_test, dict)
    overall = final_test["overall"]
    assert isinstance(overall, dict)
    overall["wape"] = None

    markdown = render_ridge_evaluation_report(summary)

    assert "| WAPE | n/a |" in markdown


def test_render_ridge_evaluation_report_rejects_missing_required_field() -> None:
    summary = deepcopy(ridge_summary())
    final_test = summary["final_test"]
    assert isinstance(final_test, dict)
    overall = final_test["overall"]
    assert isinstance(overall, dict)
    del overall["mae"]

    with pytest.raises(
        RidgeReportError,
        match="missing required summary field: final_test.overall.mae",
    ):
        render_ridge_evaluation_report(summary)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -v
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'urbanflow.modeling.reports'`.

- [ ] **Step 3: Implement the pure Markdown renderer**

Create `src/urbanflow/modeling/reports.py`:

```python
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class RidgeReportError(ValueError):
    """Raised when a Ridge evaluation summary cannot be rendered."""


def _field_path(parent: str, key: str) -> str:
    return key if not parent else f"{parent}.{key}"


def _required(mapping: Mapping[str, Any], key: str, *, path: str = "") -> Any:
    if key not in mapping:
        raise RidgeReportError(f"missing required summary field: {_field_path(path, key)}")
    return mapping[key]


def _required_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RidgeReportError(f"summary field must be an object: {path}")
    return value


def _required_sequence(value: Any, *, path: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RidgeReportError(f"summary field must be a list: {path}")
    return value


def _metric_text(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(metric):
        return "n/a"
    return f"{metric:.4f}"


def _count_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _cell_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metric_mapping(window: Mapping[str, Any], *, path: str) -> Mapping[str, Any]:
    overall_path = _field_path(path, "overall")
    overall = _required_mapping(_required(window, "overall", path=path), path=overall_path)
    for key in ("row_count", "mae", "rmse", "wape"):
        _required(overall, key, path=overall_path)
    return overall


def _window_mapping(summary: Mapping[str, Any], key: str, *, path: str = "") -> Mapping[str, Any]:
    window_path = _field_path(path, key)
    window = _required_mapping(_required(summary, key, path=path), path=window_path)
    for field in ("name", "start", "end", "train_end", "training_row_count", "horizon_metrics"):
        _required(window, field, path=window_path)
    _metric_mapping(window, path=window_path)
    return window


def _horizon_records(window: Mapping[str, Any], *, path: str) -> Sequence[Any]:
    records_path = _field_path(path, "horizon_metrics")
    records = _required_sequence(_required(window, "horizon_metrics", path=path), path=records_path)
    for index, record in enumerate(records):
        record_path = f"{records_path}.{index}"
        mapping = _required_mapping(record, path=record_path)
        for field in ("forecast_horizon", "row_count", "mae", "rmse", "wape"):
            _required(mapping, field, path=record_path)
    return records


def _period(window: Mapping[str, Any]) -> str:
    return f"{_cell_text(window['start'])} to {_cell_text(window['end'])}"


def _validation_row(window: Mapping[str, Any]) -> str:
    overall = _metric_mapping(window, path=str(window["name"]))
    return (
        f"| {_cell_text(window['name'])} | {_period(window)} | "
        f"{_count_text(window['training_row_count'])} | "
        f"{_count_text(overall['row_count'])} | "
        f"{_metric_text(overall['mae'])} | "
        f"{_metric_text(overall['rmse'])} | "
        f"{_metric_text(overall['wape'])} |"
    )


def _horizon_row(record: Mapping[str, Any]) -> str:
    return (
        f"| {_count_text(record['forecast_horizon'])} | "
        f"{_count_text(record['row_count'])} | "
        f"{_metric_text(record['mae'])} | "
        f"{_metric_text(record['rmse'])} | "
        f"{_metric_text(record['wape'])} |"
    )


def render_ridge_evaluation_report(summary: Mapping[str, Any]) -> str:
    for field in (
        "input_path",
        "row_count",
        "validation_window_count",
        "validation_windows",
        "final_test",
    ):
        _required(summary, field)

    validation_windows = _required_sequence(
        summary["validation_windows"],
        path="validation_windows",
    )
    validation_window_mappings = [
        _required_mapping(window, path=f"validation_windows.{index}")
        for index, window in enumerate(validation_windows)
    ]
    for index, window in enumerate(validation_window_mappings):
        window_path = f"validation_windows.{index}"
        for field in ("name", "start", "end", "train_end", "training_row_count", "horizon_metrics"):
            _required(window, field, path=window_path)
        _metric_mapping(window, path=window_path)

    final_test = _window_mapping(summary, "final_test")
    final_overall = _metric_mapping(final_test, path="final_test")
    final_horizons = [
        _required_mapping(record, path=f"final_test.horizon_metrics.{index}")
        for index, record in enumerate(_horizon_records(final_test, path="final_test"))
    ]

    lines = [
        "# Ridge Evaluation Report",
        "",
        f"Source: `{_cell_text(summary['input_path'])}`",
        "",
        f"Rows evaluated: {_count_text(summary['row_count'])}",
        f"Validation windows: {_count_text(summary['validation_window_count'])}",
        "",
        "## Final test",
        "",
        f"Window: `{_cell_text(final_test['name'])}`",
        f"Period: {_period(final_test)}",
        f"Training rows: {_count_text(final_test['training_row_count'])}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Row count | {_count_text(final_overall['row_count'])} |",
        f"| MAE | {_metric_text(final_overall['mae'])} |",
        f"| RMSE | {_metric_text(final_overall['rmse'])} |",
        f"| WAPE | {_metric_text(final_overall['wape'])} |",
        "",
        "## Validation windows",
        "",
        "| Window | Period | Training rows | Rows | MAE | RMSE | WAPE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_validation_row(window) for window in validation_window_mappings)
    lines.extend(
        [
            "",
            "## Final test by horizon",
            "",
            "| Horizon | Rows | MAE | RMSE | WAPE |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(_horizon_row(record) for record in final_horizons)
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run renderer tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -v
```

Expected: PASS with 3 tests.

- [ ] **Step 5: Run targeted renderer quality checks**

Run:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check src/urbanflow/modeling/reports.py tests/unit/modeling/test_modeling_reports.py --no-cache
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check src/urbanflow/modeling/reports.py tests/unit/modeling/test_modeling_reports.py
```

Expected: Ruff check passes and format check passes.

- [ ] **Step 6: Commit the renderer**

Run:

```powershell
git add src/urbanflow/modeling/reports.py tests/unit/modeling/test_modeling_reports.py
git commit -m "feat: add ridge markdown report renderer"
```

Expected: one commit containing the pure Markdown renderer and renderer tests.

## Task 2: Report CLI and script wrapper

**Files:**
- Modify: `tests/unit/modeling/test_modeling_reports.py`
- Create: `src/urbanflow/modeling/report_cli.py`
- Create: `scripts/render_ridge_evaluation_report.py`

- [ ] **Step 1: Add failing report CLI tests**

First replace the import block at the top of `tests/unit/modeling/test_modeling_reports.py`
with this Ruff-compatible import block:

```python
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from urbanflow.modeling.report_cli import main as report_main
from urbanflow.modeling.reports import RidgeReportError, render_ridge_evaluation_report
```

Then append these tests to `tests/unit/modeling/test_modeling_reports.py`:

```python


def write_summary_json(tmp_path: Path) -> Path:
    path = tmp_path / "ridge_evaluation.json"
    path.write_text(json.dumps(ridge_summary()), encoding="utf-8")
    return path


def test_report_cli_writes_markdown_file(tmp_path, capsys) -> None:
    summary_path = write_summary_json(tmp_path)
    output_path = tmp_path / "reports" / "ridge_evaluation.md"

    exit_code = report_main([str(summary_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload == {"output_path": str(output_path)}
    assert "# Ridge Evaluation Report" in output_path.read_text(encoding="utf-8")


def test_report_cli_returns_two_when_output_exists_without_force(tmp_path, capsys) -> None:
    summary_path = write_summary_json(tmp_path)
    output_path = tmp_path / "ridge_evaluation.md"
    output_path.write_text("existing", encoding="utf-8")

    exit_code = report_main([str(summary_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "output file already exists" in captured.err
    assert output_path.read_text(encoding="utf-8") == "existing"


def test_report_cli_force_overwrites_existing_output(tmp_path, capsys) -> None:
    summary_path = write_summary_json(tmp_path)
    output_path = tmp_path / "ridge_evaluation.md"
    output_path.write_text("existing", encoding="utf-8")

    exit_code = report_main([str(summary_path), "--output", str(output_path), "--force"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "# Ridge Evaluation Report" in output_path.read_text(encoding="utf-8")


def test_render_ridge_evaluation_report_script_help() -> None:
    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [
            sys.executable,
            repository_root / "scripts" / "render_ridge_evaluation_report.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Render a Ridge evaluation Markdown report" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -v
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'urbanflow.modeling.report_cli'`.

- [ ] **Step 3: Implement the report CLI**

Create `src/urbanflow/modeling/report_cli.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from urbanflow.modeling.reports import RidgeReportError, render_ridge_evaluation_report


class RidgeReportCliError(ValueError):
    """Raised when the Ridge report CLI receives invalid local inputs."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a Ridge evaluation Markdown report from a JSON summary."
    )
    parser.add_argument(
        "summary_json",
        type=Path,
        help="Path to a Ridge evaluation JSON summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path. Defaults to the input path with .md suffix.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser


def _read_summary_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RidgeReportCliError(f"summary JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RidgeReportCliError(f"could not read summary JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RidgeReportCliError(f"invalid summary JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RidgeReportCliError("summary JSON must contain an object")
    return payload


def _resolve_output_path(summary_json: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return summary_json.with_suffix(".md")


def render_report_file(
    summary_json: Path,
    *,
    output_path: Path | None = None,
    force: bool = False,
) -> Path:
    destination = _resolve_output_path(summary_json, output_path)
    if destination.exists() and not force:
        raise RidgeReportCliError(f"output file already exists: {destination}")

    summary = _read_summary_json(summary_json)
    markdown = render_ridge_evaluation_report(summary)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        raise RidgeReportCliError(f"could not write report: {destination}") from exc
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = render_report_file(
            args.summary_json,
            output_path=args.output,
            force=args.force,
        )
    except (RidgeReportCliError, RidgeReportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"output_path": str(output_path)}, sort_keys=True))
    return 0
```

- [ ] **Step 4: Run CLI tests and confirm only script help still fails**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -v
```

Expected: all renderer and CLI tests pass except `test_render_ridge_evaluation_report_script_help`, which fails because `scripts/render_ridge_evaluation_report.py` does not exist.

- [ ] **Step 5: Add the script wrapper**

Create `scripts/render_ridge_evaluation_report.py`:

```python
from urbanflow.modeling.report_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run report tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -v
```

Expected: PASS with 7 tests.

- [ ] **Step 7: Run script help manually**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' scripts/render_ridge_evaluation_report.py --help
```

Expected: stdout contains `Render a Ridge evaluation Markdown report from a JSON summary.`

- [ ] **Step 8: Run targeted report quality checks**

Run:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check src/urbanflow/modeling/reports.py src/urbanflow/modeling/report_cli.py tests/unit/modeling/test_modeling_reports.py scripts/render_ridge_evaluation_report.py --no-cache
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check src/urbanflow/modeling/reports.py src/urbanflow/modeling/report_cli.py tests/unit/modeling/test_modeling_reports.py scripts/render_ridge_evaluation_report.py
```

Expected: Ruff check passes and format check passes.

- [ ] **Step 9: Commit the report CLI**

Run:

```powershell
git add src/urbanflow/modeling/report_cli.py scripts/render_ridge_evaluation_report.py tests/unit/modeling/test_modeling_reports.py
git commit -m "feat: add ridge markdown report cli"
```

Expected: one commit containing report CLI, script wrapper, and CLI tests.

## Task 3: README documentation and full verification

**Files:**
- Modify: `README.md`
- Verify: repository quality gate

- [ ] **Step 1: Update README with the JSON-to-Markdown flow**

In `README.md`, under `## Train a local Ridge baseline`, append this paragraph and command block after the existing Ridge evaluation CLI example:

````markdown
To render the JSON summary into a Markdown report, run:

```powershell
python scripts/evaluate_ridge_baseline.py data/modeling/supervised_rows.csv --validation-months 3 > reports/modeling/ridge_evaluation.json
python scripts/render_ridge_evaluation_report.py reports/modeling/ridge_evaluation.json --output reports/modeling/ridge_evaluation.md
```

The `reports/` directory is for local generated artifacts and is not required
for unit tests.
````

- [ ] **Step 2: Run focused modeling report tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py tests/unit/modeling/test_modeling_cli.py -v
```

Expected: PASS with all modeling CLI and report tests.

- [ ] **Step 3: Run targeted quality checks**

Run:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check src/urbanflow/modeling tests/unit/modeling scripts/render_ridge_evaluation_report.py --no-cache
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check src/urbanflow/modeling tests/unit/modeling scripts/render_ridge_evaluation_report.py
```

Expected: Ruff check passes and format check passes.

- [ ] **Step 4: Commit README update**

Run:

```powershell
git add README.md
git commit -m "docs: document ridge markdown report workflow"
```

Expected: one documentation commit containing only README changes.

- [ ] **Step 5: Run the full quality gate on the implementation branch**

Run:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check . --no-cache
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check .
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest
```

Expected: Ruff passes, format check passes, and all tests pass.

- [ ] **Step 6: Self-review implementation scope**

Run:

```powershell
git status --short --branch
git diff --stat main...HEAD
git log --oneline -6
rg -n "Streamlit|MLflow|LightGBM|PostgreSQL|artifact|html|pdf|chart" src/urbanflow/modeling scripts README.md tests/unit/modeling
```

Expected:

- working tree is clean on `codex/ridge-evaluation-report`;
- diff contains only `src/urbanflow/modeling/reports.py`, `src/urbanflow/modeling/report_cli.py`, `scripts/render_ridge_evaluation_report.py`, `tests/unit/modeling/test_modeling_reports.py`, and `README.md`;
- search hits are limited to README/spec-adjacent prose or no hits, with no new Streamlit, MLflow, LightGBM, PostgreSQL reader, artifact persistence, HTML/PDF renderer, or chart code.

## Task 4: Merge, push, and cleanup

**Files:**
- Merge: `codex/ridge-evaluation-report` into `main`
- Push: `main`

- [ ] **Step 1: Verify remote main has not moved**

Run from `D:\Github项目\UrbanFlow-AU`:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 fetch origin main
git rev-list --left-right --count main...origin/main
```

Expected: `0 0`.

- [ ] **Step 2: Merge to main only**

Run from `D:\Github项目\UrbanFlow-AU`:

```powershell
git merge --ff-only codex/ridge-evaluation-report
git status --short --branch
```

Expected: `main` fast-forwards and is ahead of `origin/main`.

- [ ] **Step 3: Re-run the full quality gate on main**

Run from `D:\Github项目\UrbanFlow-AU`:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check . --no-cache
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check .
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest
```

Expected: Ruff passes, format check passes, and all tests pass on `main`.

- [ ] **Step 4: Push main**

Run from `D:\Github项目\UrbanFlow-AU`:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 push origin main
```

Expected: only `main` is pushed to GitHub.

- [ ] **Step 5: Remove the local worktree and local codex branch**

Before removal, verify the resolved path stays under `D:\Github项目\UrbanFlow-AU\.worktrees`:

```powershell
$target = (Resolve-Path '.worktrees\ridge-evaluation-report').Path
$root = (Resolve-Path '.worktrees').Path
if (-not $target.StartsWith($root)) { throw "Refusing to remove worktree outside .worktrees: $target" }
git worktree remove $target
git worktree prune
git branch -d codex/ridge-evaluation-report
```

Expected: local implementation worktree and local codex branch are removed after successful merge and push.

## Self-review checklist

Before executing Task 4 merge/push:

- Spec coverage:
  - Ridge JSON summary input: Task 1 renderer tests and Task 2 CLI tests.
  - Markdown title/source/row counts: Task 1.
  - final-test overall metrics: Task 1.
  - validation-window metrics table: Task 1.
  - final-test per-horizon metrics table: Task 1.
  - missing metric formatting as `n/a`: Task 1.
  - missing required field error path: Task 1.
  - `--output` and `--force`: Task 2.
  - thin script wrapper and script help: Task 2.
  - README workflow: Task 3.
- Type consistency:
  - report renderer entrypoint is `render_ridge_evaluation_report(summary)`.
  - renderer errors use `RidgeReportError`.
  - CLI errors use `RidgeReportCliError`.
  - CLI file renderer is `render_report_file(summary_json, output_path=None, force=False)`.
  - CLI `main` returns `0` on success and `2` for expected user errors.
- Scope guard:
  - no HTML/PDF renderer;
  - no chart rendering;
  - no Streamlit/dashboard;
  - no Ridge model execution inside report command;
  - no MLflow logging;
  - no PostgreSQL reads;
  - no new dependencies.
