# Ridge Report Mermaid Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic Mermaid metric-comparison charts to the existing Ridge evaluation Markdown report.

**Architecture:** Keep the existing Markdown report renderer as the only public API. Add small private helpers in `src/urbanflow/modeling/reports.py` to derive chart points from already-validated `overall` metrics, render Mermaid `xychart-beta` blocks, and insert the chart section between validation-window metrics and final-test horizon metrics. Preserve existing tables as the exact-value fallback.

**Tech Stack:** Python 3.12, standard library only, Markdown, Mermaid fenced code blocks, pytest, Ruff.

---

## Scope and source design

Use this spec as the implementation contract:

- `docs/superpowers/specs/2026-07-03-ridge-report-mermaid-design.md`

This plan implements only the Mermaid-in-Markdown enhancement described there. It does not add HTML, PDF, image export, dashboards, new model metrics, new dependencies, or changes to Ridge evaluation logic.

## File map

- Modify `src/urbanflow/modeling/reports.py`
  - Add private constants and helpers for chart metric keys, numeric extraction, axis bounds, Mermaid label formatting, chart-point derivation, chart-block rendering, and chart-section rendering.
  - Insert chart-section lines in `render_ridge_evaluation_report(summary)` after validation-window rows and before `## Final test by horizon`.
- Modify `tests/unit/modeling/test_modeling_reports.py`
  - Add renderer tests for chart section placement and exact Mermaid blocks.
  - Add renderer tests for nonnumeric metric skipping and Mermaid label escaping.
  - Keep existing CLI and example drift tests.
- Modify `docs/examples/modeling/ridge_evaluation_report.md`
  - Regenerate from `docs/examples/modeling/ridge_evaluation_summary.json` after renderer changes.
- Modify `README.md`
  - Document that Markdown reports include Mermaid charts when supported and tables remain the exact-value fallback.

## Task 1: Render basic Mermaid metric charts

**Files:**

- Modify: `tests/unit/modeling/test_modeling_reports.py`
- Modify: `src/urbanflow/modeling/reports.py`
- Modify: `docs/examples/modeling/ridge_evaluation_report.md`

- [ ] **Step 1: Write the failing chart-section test**

Add this test after `test_render_ridge_evaluation_report_includes_core_sections` in `tests/unit/modeling/test_modeling_reports.py`:

```python
def test_render_ridge_evaluation_report_includes_mermaid_metric_charts() -> None:
    markdown = render_ridge_evaluation_report(ridge_summary())

    assert markdown.index("## Validation windows") < markdown.index(
        "## Metric comparison charts"
    )
    assert markdown.index("## Metric comparison charts") < markdown.index(
        "## Final test by horizon"
    )
    assert markdown.count("```mermaid\nxychart-beta") == 3

    expected_mae_chart = """```mermaid
xychart-beta
    title "MAE by evaluation window"
    x-axis ["validation_2025-01", "final_test_2025-02"]
    y-axis "MAE" 0 --> 1.3580
    bar [1.2346, 1.2000]
```"""
    expected_rmse_chart = """```mermaid
xychart-beta
    title "RMSE by evaluation window"
    x-axis ["validation_2025-01", "final_test_2025-02"]
    y-axis "RMSE" 0 --> 1.9298
    bar [1.7543, 1.7000]
```"""
    expected_wape_chart = """```mermaid
xychart-beta
    title "WAPE by evaluation window"
    x-axis ["validation_2025-01", "final_test_2025-02"]
    y-axis "WAPE" 0 --> 0.0894
    bar [0.0812, 0.0700]
```"""

    assert expected_mae_chart in markdown
    assert expected_rmse_chart in markdown
    assert expected_wape_chart in markdown
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py::test_render_ridge_evaluation_report_includes_mermaid_metric_charts -q
```

Expected result: the test fails because `## Metric comparison charts` is not in the rendered report.

- [ ] **Step 3: Add the minimal chart-rendering helpers**

In `src/urbanflow/modeling/reports.py`, add this constant after `RidgeReportError`:

```python
_METRIC_CHARTS = (
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("wape", "WAPE"),
)
```

Add these helpers after `_horizon_row`:

```python
def _numeric_metric_value(value: Any) -> float | None:
    if value is None:
        return None
    metric = float(value)
    if math.isnan(metric):
        return None
    return metric


def _mermaid_label(value: Any) -> str:
    text = str(value).replace("\n", " ")
    return f'"{text}"'


def _chart_axis_upper_bound(values: Sequence[float]) -> float:
    if max(values) == 0:
        return 1.0
    return max(values) * 1.1


def _metric_chart_points(
    validation_windows: Sequence[Mapping[str, Any]],
    final_test: Mapping[str, Any],
    metric_key: str,
) -> list[tuple[str, float]]:
    points: list[tuple[str, float]] = []
    for window in [*validation_windows, final_test]:
        overall = _metric_mapping(window, path=str(window["name"]))
        metric = _numeric_metric_value(overall[metric_key])
        if metric is None:
            continue
        points.append((str(window["name"]), metric))
    return points


def _mermaid_metric_chart(metric_label: str, points: Sequence[tuple[str, float]]) -> str:
    labels = ", ".join(_mermaid_label(label) for label, _ in points)
    values = ", ".join(f"{value:.4f}" for _, value in points)
    upper_bound = _chart_axis_upper_bound([value for _, value in points])
    return "\n".join(
        [
            "```mermaid",
            "xychart-beta",
            f'    title "{metric_label} by evaluation window"',
            f"    x-axis [{labels}]",
            f'    y-axis "{metric_label}" 0 --> {upper_bound:.4f}',
            f"    bar [{values}]",
            "```",
        ]
    )


def _metric_comparison_chart_lines(
    validation_windows: Sequence[Mapping[str, Any]],
    final_test: Mapping[str, Any],
) -> list[str]:
    chart_blocks = []
    for metric_key, metric_label in _METRIC_CHARTS:
        points = _metric_chart_points(validation_windows, final_test, metric_key)
        if not points:
            continue
        chart_blocks.append(_mermaid_metric_chart(metric_label, points))

    if not chart_blocks:
        return []

    lines = ["", "## Metric comparison charts", ""]
    for index, chart_block in enumerate(chart_blocks):
        if index > 0:
            lines.append("")
        lines.extend(chart_block.splitlines())
    return lines
```

Then replace this block in `render_ridge_evaluation_report(summary)`:

```python
    lines.extend(_validation_row(window) for window in validation_window_mappings)
    lines.extend(
        [
            "",
            "## Final test by horizon",
```

with this block:

```python
    lines.extend(_validation_row(window) for window in validation_window_mappings)
    lines.extend(_metric_comparison_chart_lines(validation_window_mappings, final_test))
    lines.extend(
        [
            "",
            "## Final test by horizon",
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py::test_render_ridge_evaluation_report_includes_mermaid_metric_charts -q
```

Expected result: `1 passed`.

- [ ] **Step 5: Regenerate the checked-in synthetic report**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' scripts/render_ridge_evaluation_report.py docs/examples/modeling/ridge_evaluation_summary.json --output docs/examples/modeling/ridge_evaluation_report.md --force
```

Expected result: stdout contains JSON with output path `docs/examples/modeling/ridge_evaluation_report.md`.

- [ ] **Step 6: Run the modeling report tests**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -q
```

Expected result: all modeling report tests pass, including the checked-in example drift test.

- [ ] **Step 7: Commit the basic chart renderer**

Run:

```powershell
git add src/urbanflow/modeling/reports.py tests/unit/modeling/test_modeling_reports.py docs/examples/modeling/ridge_evaluation_report.md
git commit -m "feat: add ridge report mermaid charts"
```

## Task 2: Skip nonnumeric Mermaid values without changing table output

**Files:**

- Modify: `tests/unit/modeling/test_modeling_reports.py`
- Modify: `src/urbanflow/modeling/reports.py`

- [ ] **Step 1: Write the failing nonnumeric-metric test**

Add this test after `test_render_ridge_evaluation_report_formats_missing_metrics_as_na`:

```python
def test_render_ridge_evaluation_report_omits_nonnumeric_metrics_from_mermaid_charts() -> None:
    summary = deepcopy(ridge_summary())
    validation_windows = summary["validation_windows"]
    assert isinstance(validation_windows, list)
    validation_window = validation_windows[0]
    assert isinstance(validation_window, dict)
    validation_overall = validation_window["overall"]
    assert isinstance(validation_overall, dict)
    validation_overall["mae"] = None

    final_test = summary["final_test"]
    assert isinstance(final_test, dict)
    final_overall = final_test["overall"]
    assert isinstance(final_overall, dict)
    final_overall["mae"] = "unavailable"

    markdown = render_ridge_evaluation_report(summary)

    assert "| MAE | unavailable |" in markdown
    assert (
        "| validation_2025-01 | 2025-01-01T00:00:00+11:00 to "
        "2025-02-01T00:00:00+11:00 | 744 | 744 | n/a | 1.7543 | 0.0812 |"
    ) in markdown
    assert 'title "MAE by evaluation window"' not in markdown
    assert 'title "RMSE by evaluation window"' in markdown
    assert 'title "WAPE by evaluation window"' in markdown
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py::test_render_ridge_evaluation_report_omits_nonnumeric_metrics_from_mermaid_charts -q
```

Expected result: the test fails with `ValueError` from `float("unavailable")`.

- [ ] **Step 3: Make numeric extraction tolerant**

Replace `_numeric_metric_value` in `src/urbanflow/modeling/reports.py` with:

```python
def _numeric_metric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(metric) or math.isinf(metric):
        return None
    return metric
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py::test_render_ridge_evaluation_report_omits_nonnumeric_metrics_from_mermaid_charts -q
```

Expected result: `1 passed`.

- [ ] **Step 5: Run the modeling report tests**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -q
```

Expected result: all modeling report tests pass.

- [ ] **Step 6: Commit the nonnumeric skip behavior**

Run:

```powershell
git add src/urbanflow/modeling/reports.py tests/unit/modeling/test_modeling_reports.py
git commit -m "fix: skip nonnumeric ridge report chart metrics"
```

## Task 3: Escape Mermaid labels deterministically

**Files:**

- Modify: `tests/unit/modeling/test_modeling_reports.py`
- Modify: `src/urbanflow/modeling/reports.py`

- [ ] **Step 1: Write the failing Mermaid-label escaping test**

Add this test after `test_render_ridge_evaluation_report_includes_mermaid_metric_charts`:

```python
def test_render_ridge_evaluation_report_escapes_mermaid_window_labels() -> None:
    summary = deepcopy(ridge_summary())
    validation_windows = summary["validation_windows"]
    assert isinstance(validation_windows, list)
    validation_window = validation_windows[0]
    assert isinstance(validation_window, dict)
    validation_window["name"] = 'validation "quoted"\nwindow'

    markdown = render_ridge_evaluation_report(summary)

    assert 'x-axis ["validation \\"quoted\\" window", "final_test_2025-02"]' in markdown
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py::test_render_ridge_evaluation_report_escapes_mermaid_window_labels -q
```

Expected result: the test fails because `_mermaid_label` does not escape double quotes.

- [ ] **Step 3: Use JSON string encoding for Mermaid labels**

Add `json` to the imports at the top of `src/urbanflow/modeling/reports.py`:

```python
import json
import math
```

Replace `_mermaid_label` with:

```python
def _mermaid_label(value: Any) -> str:
    text = str(value).replace("\n", " ")
    return json.dumps(text, ensure_ascii=False)
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py::test_render_ridge_evaluation_report_escapes_mermaid_window_labels -q
```

Expected result: `1 passed`.

- [ ] **Step 5: Run the modeling report tests**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -q
```

Expected result: all modeling report tests pass.

- [ ] **Step 6: Commit the Mermaid label escaping**

Run:

```powershell
git add src/urbanflow/modeling/reports.py tests/unit/modeling/test_modeling_reports.py
git commit -m "fix: escape ridge report mermaid labels"
```

## Task 4: Document the visual layer

**Files:**

- Modify: `README.md`
- Test: `tests/unit/modeling/test_modeling_reports.py`

- [ ] **Step 1: Confirm the regenerated example includes the expected Mermaid section**

Run:

```powershell
Select-String -Path docs/examples/modeling/ridge_evaluation_report.md -Pattern 'Metric comparison charts','xychart-beta','MAE by evaluation window','RMSE by evaluation window','WAPE by evaluation window'
```

Expected result: each pattern is present at least once.

- [ ] **Step 2: Update the README report guidance**

In `README.md`, after the checked-in synthetic example link, add this paragraph:

```markdown
The generated Markdown report includes exact metric tables plus Mermaid
comparison charts for viewers that support Mermaid, such as GitHub. If a viewer
does not render Mermaid charts, the tables remain the source of exact values.
```

- [ ] **Step 3: Run the modeling report tests and verify the example drift test passes**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -q
```

Expected result: all modeling report tests pass.

- [ ] **Step 4: Commit the README update**

Run:

```powershell
git add README.md
git commit -m "docs: document ridge report mermaid charts"
```

## Task 5: Final verification before merge

**Files:**

- Verify the full repository.

- [ ] **Step 1: Run Ruff lint**

Run:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' check . --no-cache
```

Expected result:

```text
All checks passed!
```

- [ ] **Step 2: Run Ruff format check**

Run:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\ruff.exe' format --check .
```

Expected result:

```text
92 files already formatted
```

If the file count changes because Python files are added or removed before execution, accept the new count only if the command exits `0`.

- [ ] **Step 3: Run the full pytest suite**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest
```

Expected result: all tests pass. At plan-writing time, the suite has 144 tests; after implementation it should increase by the new report tests.

- [ ] **Step 4: Inspect the final diff**

Run:

```powershell
git status --short
git diff --check
git log --oneline -5
```

Expected result:

- `git status --short` shows no uncommitted files.
- `git diff --check` prints no whitespace errors.
- recent commits include:
  - `feat: add ridge report mermaid charts`
  - `fix: skip nonnumeric ridge report chart metrics`
  - `fix: escape ridge report mermaid labels`
  - `docs: document ridge report mermaid charts`

## Implementation self-review checklist

Before merging to `main`, verify these requirements against the diff:

- The chart section title is exactly `## Metric comparison charts`.
- Chart section placement is after `## Validation windows` and before `## Final test by horizon`.
- Metric chart order is MAE, RMSE, WAPE.
- Window order is all validation windows in existing order, followed by final test.
- Chart values use four decimal places.
- Y-axis upper bound is 110% of the largest included value, or `1.0000` when all included values are zero.
- `None`, `NaN`, infinity, and nonnumeric metric values are omitted from Mermaid charts.
- Tables still render exact values and `n/a` as before.
- Mermaid labels collapse newlines and escape double quotes through JSON string encoding.
- The checked-in example report matches the renderer output.
- README explains Mermaid support as an enhancement and tables as the exact-value fallback.
