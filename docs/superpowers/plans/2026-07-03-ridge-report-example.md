# Ridge Report Example Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a checked-in synthetic Ridge evaluation example JSON and Markdown report that users can inspect without running local model commands.

**Architecture:** Keep the example under `docs/examples/modeling/` so it is repository-visible and not confused with local generated `reports/` output. Add one unit test that renders the example JSON with the existing Ridge Markdown renderer and compares it exactly to the checked-in Markdown report. Add a README link that labels the example as synthetic.

**Tech Stack:** Python 3.11+, standard-library `json`/`pathlib`, existing `urbanflow.modeling.reports.render_ridge_evaluation_report`, pytest, Ruff, Markdown.

---

## Source spec

Implement:

`docs/superpowers/specs/2026-07-02-ridge-report-example-design.md`

## Worktree and execution note

Create an isolated implementation worktree before executing this plan:

```powershell
git worktree add '.worktrees/ridge-report-example' -b codex/ridge-report-example
cd '.worktrees/ridge-report-example'
$env:PYTHONPATH='src'
```

Use the existing project virtual environment:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest
```

When finishing, merge local `codex/ridge-report-example` into `main` and push
only `main`.

## File structure

- Create `docs/examples/modeling/ridge_evaluation_summary.json`
  - Stores a compact synthetic Ridge evaluation summary matching the report renderer input contract.
- Create `docs/examples/modeling/ridge_evaluation_report.md`
  - Stores the Markdown output expected from `render_ridge_evaluation_report`.
- Modify `tests/unit/modeling/test_modeling_reports.py`
  - Adds one repository-file drift test that renders the JSON and compares it to the Markdown.
- Modify `README.md`
  - Links to the checked-in synthetic example report from the Ridge baseline section.

## Task 1: Example artifact and drift test

**Files:**
- Create: `docs/examples/modeling/ridge_evaluation_summary.json`
- Create: `docs/examples/modeling/ridge_evaluation_report.md`
- Modify: `tests/unit/modeling/test_modeling_reports.py`

- [ ] **Step 1: Write the failing drift test**

Append this test to `tests/unit/modeling/test_modeling_reports.py`:

```python


def test_checked_in_ridge_example_report_matches_renderer() -> None:
    repository_root = Path(__file__).parents[3]
    summary_path = (
        repository_root
        / "docs"
        / "examples"
        / "modeling"
        / "ridge_evaluation_summary.json"
    )
    report_path = (
        repository_root
        / "docs"
        / "examples"
        / "modeling"
        / "ridge_evaluation_report.md"
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert render_ridge_evaluation_report(summary) == report_path.read_text(encoding="utf-8")
```

The current test module already imports `json`, `Path`, and
`render_ridge_evaluation_report`, so no import changes are needed.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py::test_checked_in_ridge_example_report_matches_renderer -v
```

Expected: FAIL with `FileNotFoundError` for
`docs/examples/modeling/ridge_evaluation_summary.json`.

- [ ] **Step 3: Add the synthetic example JSON**

Create `docs/examples/modeling/ridge_evaluation_summary.json`:

```json
{
  "final_test": {
    "end": "2025-03-01T00:00:00+11:00",
    "horizon_metrics": [
      {
        "forecast_horizon": 1,
        "mae": 1.1,
        "rmse": 1.6,
        "row_count": 336,
        "wape": 0.06
      },
      {
        "forecast_horizon": 24,
        "mae": 1.3,
        "rmse": 1.8,
        "row_count": 336,
        "wape": 0.08
      }
    ],
    "name": "final_test_2025-02",
    "overall": {
      "mae": 1.2,
      "rmse": 1.7,
      "row_count": 672,
      "wape": 0.07
    },
    "start": "2025-02-01T00:00:00+11:00",
    "train_end": "2025-02-01T00:00:00+11:00",
    "training_row_count": 1488
  },
  "input_path": "docs/examples/modeling/synthetic_supervised_rows.csv",
  "row_count": 1464,
  "validation_window_count": 1,
  "validation_windows": [
    {
      "end": "2025-02-01T00:00:00+11:00",
      "horizon_metrics": [
        {
          "forecast_horizon": 1,
          "mae": 1.2345,
          "rmse": 1.7654,
          "row_count": 744,
          "wape": 0.0812
        }
      ],
      "name": "validation_2025-01",
      "overall": {
        "mae": 1.2345,
        "rmse": 1.7654,
        "row_count": 744,
        "wape": 0.0812
      },
      "start": "2025-01-01T00:00:00+11:00",
      "train_end": "2025-01-01T00:00:00+11:00",
      "training_row_count": 744
    }
  ]
}
```

- [ ] **Step 4: Add the rendered Markdown example**

Create `docs/examples/modeling/ridge_evaluation_report.md`:

```markdown
# Ridge Evaluation Report

Source: `docs/examples/modeling/synthetic_supervised_rows.csv`

Rows evaluated: 1464
Validation windows: 1

## Final test

Window: `final_test_2025-02`
Period: 2025-02-01T00:00:00+11:00 to 2025-03-01T00:00:00+11:00
Training rows: 1488

| Metric | Value |
| --- | ---: |
| Row count | 672 |
| MAE | 1.2000 |
| RMSE | 1.7000 |
| WAPE | 0.0700 |

## Validation windows

| Window | Period | Training rows | Rows | MAE | RMSE | WAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation_2025-01 | 2025-01-01T00:00:00+11:00 to 2025-02-01T00:00:00+11:00 | 744 | 744 | 1.2345 | 1.7654 | 0.0812 |

## Final test by horizon

| Horizon | Rows | MAE | RMSE | WAPE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 336 | 1.1000 | 1.6000 | 0.0600 |
| 24 | 336 | 1.3000 | 1.8000 | 0.0800 |
```

Ensure the file ends with one trailing newline.

- [ ] **Step 5: Run the drift test and report tests**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -v
```

Expected: PASS with 8 report tests.

- [ ] **Step 6: Run targeted quality checks**

Run:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check tests/unit/modeling/test_modeling_reports.py --no-cache
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check tests/unit/modeling/test_modeling_reports.py
```

Expected: Ruff check passes and format check passes.

- [ ] **Step 7: Commit the example artifact**

Run:

```powershell
git add docs/examples/modeling/ridge_evaluation_summary.json docs/examples/modeling/ridge_evaluation_report.md tests/unit/modeling/test_modeling_reports.py
git commit -m "docs: add ridge evaluation report example"
```

Expected: one commit containing the synthetic JSON, rendered Markdown, and drift test.

## Task 2: README link

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the example link**

In `README.md`, under `## Train a local Ridge baseline`, append this sentence
after the paragraph that says `reports/` is local generated output:

```markdown
A checked-in synthetic example report is available at
[`docs/examples/modeling/ridge_evaluation_report.md`](docs/examples/modeling/ridge_evaluation_report.md).
```

- [ ] **Step 2: Run focused tests and quality checks**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py tests/unit/modeling/test_modeling_cli.py -v
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check README.md tests/unit/modeling/test_modeling_reports.py --no-cache
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check README.md tests/unit/modeling/test_modeling_reports.py
```

Expected: modeling report and CLI tests pass, Ruff check passes, and format check passes.

- [ ] **Step 3: Commit the README link**

Run:

```powershell
git add README.md
git commit -m "docs: link ridge report example"
```

Expected: one documentation commit containing only the README link.

## Task 3: Full verification, merge, push, and cleanup

**Files:**
- Verify: repository quality gate
- Merge: `codex/ridge-report-example` into `main`
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
git diff --name-only main...HEAD
rg -n "Streamlit|MLflow|LightGBM|PostgreSQL|artifact persistence|html|pdf|chart|real training data" docs/examples README.md tests/unit/modeling
```

Expected:

- working tree is clean on `codex/ridge-report-example`;
- diff contains only `docs/examples/modeling/ridge_evaluation_summary.json`, `docs/examples/modeling/ridge_evaluation_report.md`, `tests/unit/modeling/test_modeling_reports.py`, and `README.md`;
- search hits are limited to README/spec-adjacent prose or no hits, with no new Streamlit, MLflow, LightGBM, PostgreSQL reader, artifact persistence, HTML/PDF renderer, chart code, or real training data.

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
git merge --ff-only codex/ridge-report-example
git status --short --branch
```

Expected: `main` fast-forwards and is ahead of `origin/main`.

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

Before removal, verify the resolved path stays under
`D:\Github项目\UrbanFlow-AU\.worktrees`:

```powershell
$target = (Resolve-Path '.worktrees\ridge-report-example').Path
$root = (Resolve-Path '.worktrees').Path
if (-not $target.StartsWith($root)) { throw "Refusing to remove worktree outside .worktrees: $target" }
git worktree remove $target
git worktree prune
git branch -d codex/ridge-report-example
```

Expected: local implementation worktree and local codex branch are removed after successful merge and push.

## Self-review checklist

Before executing Task 3 merge/push:

- Spec coverage:
  - checked-in JSON summary: Task 1.
  - checked-in Markdown report: Task 1.
  - exact renderer sync test: Task 1.
  - README link labeling the file as synthetic: Task 2.
  - no local `reports/` generated output: Task 1 and Task 3 scope review.
- Type and path consistency:
  - example JSON path is `docs/examples/modeling/ridge_evaluation_summary.json`.
  - example Markdown path is `docs/examples/modeling/ridge_evaluation_report.md`.
  - renderer entrypoint is `render_ridge_evaluation_report(summary)`.
  - implementation branch is `codex/ridge-report-example`.
- Scope guard:
  - no real training data;
  - no model fitting in tests;
  - no chart, HTML, PDF, screenshot, or image asset;
  - no Streamlit/dashboard;
  - no MLflow logging;
  - no PostgreSQL reads;
  - no new dependencies.
