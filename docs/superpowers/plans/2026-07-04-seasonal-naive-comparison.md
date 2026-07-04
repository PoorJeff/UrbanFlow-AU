# Seasonal Naive Comparison Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Seasonal Naive baseline metrics beside Ridge metrics in the same rolling-origin evaluation JSON and Markdown report.

**Architecture:** Reuse the existing supervised CSV and rolling-origin splits. Derive a compact one-row-per-location-hour panel from the supervised rows, add Seasonal Naive predictions to the same evaluation rows used by Ridge, and expose the comparison through existing JSON and Markdown report flows. Keep Ridge as the trainable model and make Seasonal Naive a transparent comparison baseline.

**Tech Stack:** Python 3.12, pandas, pytest, Ruff, existing UrbanFlow modeling modules, Markdown.

---

## Scope and source design

Use this spec as the implementation contract:

- `docs/superpowers/specs/2026-07-04-seasonal-naive-comparison-design.md`

This plan implements only the Ridge-versus-Seasonal-Naive comparison slice. It does not add LightGBM, MLflow, model artifact persistence, API fallback behavior, Streamlit changes, or new dependencies.

## File map

- Modify `src/urbanflow/modeling/baselines.py`
  - Add a small exception class for Seasonal Naive input problems.
  - Add `derive_seasonal_naive_panel(supervised_frame)` to create a compact panel from `location_id`, `target_observed_at`, and `target`.
- Modify `src/urbanflow/modeling/evaluation.py`
  - Add `ModelComparisonMetrics`.
  - Add Seasonal Naive metrics to `ModelWindowEvaluation`.
  - Add Seasonal Naive predictions to each evaluated window.
- Modify `src/urbanflow/modeling/cli.py`
  - Catch Seasonal Naive input errors as expected CLI errors.
  - Add Seasonal Naive and comparison fields to JSON summaries.
  - Reject CLI runs where Seasonal Naive has zero metric rows across all evaluated windows.
- Modify `src/urbanflow/modeling/reports.py`
  - Render `## Model comparison` only when Seasonal Naive fields are present.
  - Preserve backward compatibility for existing Ridge-only JSON.
- Modify tests:
  - `tests/unit/modeling/test_baselines.py`
  - `tests/unit/modeling/test_evaluation.py`
  - `tests/unit/modeling/test_modeling_cli.py`
  - `tests/unit/modeling/test_modeling_reports.py`
- Modify examples/docs:
  - `docs/examples/modeling/ridge_evaluation_summary.json`
  - `docs/examples/modeling/ridge_evaluation_report.md`
  - `README.md`

## Task 1: Derive a Seasonal Naive panel from supervised rows

**Files:**

- Modify: `tests/unit/modeling/test_baselines.py`
- Modify: `src/urbanflow/modeling/baselines.py`

- [ ] **Step 1: Write failing tests for panel derivation**

Append these tests to `tests/unit/modeling/test_baselines.py`:

```python
import pytest

from urbanflow.modeling.baselines import (
    SeasonalNaiveBaselineError,
    derive_seasonal_naive_panel,
)


def test_derive_seasonal_naive_panel_deduplicates_matching_targets() -> None:
    timestamps = pd.to_datetime(
        [
            "2025-01-01 00:00",
            "2025-01-01 00:00",
            "2025-01-01 01:00",
        ],
        utc=True,
    )
    supervised = pd.DataFrame(
        {
            "location_id": [101, 101, 101],
            "target_observed_at": timestamps,
            "target": [10.0, 10.0, 12.0],
        }
    )

    panel = derive_seasonal_naive_panel(supervised)

    assert list(panel.columns) == ["location_id", "observed_at", "pedestrian_count"]
    assert len(panel) == 2
    assert panel.to_dict(orient="records") == [
        {
            "location_id": 101,
            "observed_at": timestamps[0],
            "pedestrian_count": 10.0,
        },
        {
            "location_id": 101,
            "observed_at": timestamps[2],
            "pedestrian_count": 12.0,
        },
    ]


def test_derive_seasonal_naive_panel_rejects_conflicting_duplicate_targets() -> None:
    timestamp = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    supervised = pd.DataFrame(
        {
            "location_id": [101, 101],
            "target_observed_at": [timestamp, timestamp],
            "target": [10.0, 11.0],
        }
    )

    with pytest.raises(
        SeasonalNaiveBaselineError,
        match="conflicting target values for duplicate location_id and target_observed_at",
    ):
        derive_seasonal_naive_panel(supervised)


def test_derive_seasonal_naive_panel_requires_input_columns() -> None:
    supervised = pd.DataFrame({"location_id": [101], "target": [10.0]})

    with pytest.raises(
        SeasonalNaiveBaselineError,
        match="missing required Seasonal Naive input columns: target_observed_at",
    ):
        derive_seasonal_naive_panel(supervised)
```

- [ ] **Step 2: Run the new panel-derivation tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_baselines.py::test_derive_seasonal_naive_panel_deduplicates_matching_targets tests/unit/modeling/test_baselines.py::test_derive_seasonal_naive_panel_rejects_conflicting_duplicate_targets tests/unit/modeling/test_baselines.py::test_derive_seasonal_naive_panel_requires_input_columns -q
```

Expected result: import or attribute failures because the new error and helper do not exist yet.

- [ ] **Step 3: Implement panel derivation**

In `src/urbanflow/modeling/baselines.py`, add this class after imports:

```python
class SeasonalNaiveBaselineError(ValueError):
    """Raised when Seasonal Naive baseline inputs cannot be derived."""
```

Add this helper above `add_seasonal_naive_predictions`:

```python
def derive_seasonal_naive_panel(supervised_frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"location_id", "target_observed_at", "target"}
    missing_columns = sorted(required_columns.difference(supervised_frame.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise SeasonalNaiveBaselineError(
            f"missing required Seasonal Naive input columns: {missing_text}"
        )

    source = supervised_frame[["location_id", "target_observed_at", "target"]].copy()
    conflicting = (
        source.groupby(["location_id", "target_observed_at"], dropna=False)["target"]
        .nunique(dropna=False)
        .reset_index(name="target_count")
    )
    if (conflicting["target_count"] > 1).any():
        raise SeasonalNaiveBaselineError(
            "conflicting target values for duplicate location_id and target_observed_at"
        )

    panel = (
        source.drop_duplicates(subset=["location_id", "target_observed_at"])
        .rename(
            columns={
                "target_observed_at": "observed_at",
                "target": "pedestrian_count",
            }
        )
        .sort_values(["location_id", "observed_at"])
        .reset_index(drop=True)
    )
    return panel
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_baselines.py -q
```

Expected result: all baseline tests pass.

- [ ] **Step 5: Commit the panel derivation slice**

Run:

```powershell
git add src/urbanflow/modeling/baselines.py tests/unit/modeling/test_baselines.py
git commit -m "feat: derive seasonal naive panel from supervised rows"
```

## Task 2: Add Seasonal Naive metrics to rolling-origin evaluation

**Files:**

- Modify: `tests/unit/modeling/test_evaluation.py`
- Modify: `src/urbanflow/modeling/evaluation.py`

- [ ] **Step 1: Write failing evaluation tests**

In `tests/unit/modeling/test_evaluation.py`, replace `supervised_rows()` with a version that includes one week of history before the current evaluation window:

```python
def supervised_rows() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-12-25 00:00",
        "2025-01-01 07:00",
        freq="h",
        tz="Australia/Melbourne",
    )
    values = [80.0 + float(index % 24) for index in range(len(timestamps))]
    return pd.DataFrame(
        {
            "location_id": [101] * len(timestamps),
            "forecast_origin_at": timestamps - pd.Timedelta(hours=1),
            "forecast_horizon": [1 if index % 2 == 0 else 2 for index in range(len(timestamps))],
            "target_observed_at": timestamps,
            "target": values,
            "target_missing": [False] * len(timestamps),
            "pedestrian_count": [value - 5.0 for value in values],
            "pedestrian_count_missing": [False] * len(timestamps),
            "lag_1": [value - 5.0 for value in values],
            "lag_24": [value - 10.0 for value in values],
            "lag_168": [value - 20.0 for value in values],
            "rolling_24_mean": [value - 7.0 for value in values],
            "rolling_24_std": [3.0] * len(timestamps),
            "rolling_168_mean": [value - 15.0 for value in values],
            "rolling_168_std": [6.0] * len(timestamps),
            "hour": timestamps.hour,
            "weekday": timestamps.weekday,
            "month": timestamps.month,
            "is_weekend": [timestamp.weekday() >= 5 for timestamp in timestamps],
            "is_public_holiday": [False] * len(timestamps),
            "hour_sin": [0.1] * len(timestamps),
            "hour_cos": [0.9] * len(timestamps),
            "weekday_sin": [0.4] * len(timestamps),
            "weekday_cos": [0.5] * len(timestamps),
            "temperature": [20.0] * len(timestamps),
            "temperature_missing": [False] * len(timestamps),
            "rainfall": [0.0] * len(timestamps),
            "rainfall_missing": [False] * len(timestamps),
            "wind_speed": [12.0] * len(timestamps),
            "wind_speed_missing": [False] * len(timestamps),
        }
    )
```

Keep `evaluation_window()` as the 2025-01-01 04:00 to 08:00 window.

In `test_evaluate_model_window_filters_train_and_evaluation_rows`, update the
training-row assertion because the new fixture includes one week of prior
history:

```python
    assert result.model.training_row_count == 172
```

Add this test after `test_evaluate_model_window_returns_per_horizon_metrics`:

```python
def test_evaluate_model_window_returns_seasonal_naive_metrics() -> None:
    result = evaluate_model_window(supervised_rows(), evaluation_window())

    assert "seasonal_naive_prediction" in result.predictions.columns
    assert result.seasonal_naive_overall_metrics.row_count == 4
    assert set(result.seasonal_naive_horizon_metrics.columns) == {
        "forecast_horizon",
        "row_count",
        "mae",
        "rmse",
        "wape",
    }
    assert set(result.seasonal_naive_horizon_metrics["forecast_horizon"]) == {1, 2}
    assert result.model_comparison.ridge_wape == result.overall_metrics.wape
    assert (
        result.model_comparison.seasonal_naive_wape
        == result.seasonal_naive_overall_metrics.wape
    )
```

Add this helper assertion to `test_evaluate_rolling_origin_ridge_evaluates_validation_and_final_windows`:

```python
    assert result.validation_windows[0].seasonal_naive_overall_metrics.row_count > 0
    assert result.final_test.seasonal_naive_overall_metrics.row_count > 0
```

- [ ] **Step 2: Run the new evaluation test and verify it fails**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_evaluation.py::test_evaluate_model_window_returns_seasonal_naive_metrics -q
```

Expected result: failure because `ModelWindowEvaluation` has no Seasonal Naive metric fields yet.

- [ ] **Step 3: Implement Seasonal Naive evaluation fields**

In `src/urbanflow/modeling/evaluation.py`, import the baseline helpers:

```python
from urbanflow.modeling.baselines import (
    add_seasonal_naive_predictions,
    derive_seasonal_naive_panel,
)
```

Add this dataclass before `ModelWindowEvaluation`:

```python
@dataclass(frozen=True)
class ModelComparisonMetrics:
    ridge_wape: float | None
    seasonal_naive_wape: float | None
    relative_wape_improvement: float | None
```

Extend `ModelWindowEvaluation`:

```python
@dataclass(frozen=True)
class ModelWindowEvaluation:
    window: EvaluationWindow
    predictions: pd.DataFrame
    overall_metrics: RegressionMetrics
    horizon_metrics: pd.DataFrame
    model: FittedRidgeModel
    seasonal_naive_overall_metrics: RegressionMetrics
    seasonal_naive_horizon_metrics: pd.DataFrame
    model_comparison: ModelComparisonMetrics
```

Add this helper after `_evaluation_rows_for_window`:

```python
def _comparison_metrics(
    ridge_metrics: RegressionMetrics,
    seasonal_naive_metrics: RegressionMetrics,
) -> ModelComparisonMetrics:
    ridge_wape = ridge_metrics.wape
    seasonal_naive_wape = seasonal_naive_metrics.wape
    if ridge_wape is None or seasonal_naive_wape in (None, 0):
        improvement = None
    else:
        improvement = (seasonal_naive_wape - ridge_wape) / seasonal_naive_wape
    return ModelComparisonMetrics(
        ridge_wape=ridge_wape,
        seasonal_naive_wape=seasonal_naive_wape,
        relative_wape_improvement=improvement,
    )
```

Change `evaluate_model_window` signature:

```python
def evaluate_model_window(
    supervised_frame: pd.DataFrame,
    window: EvaluationWindow,
    *,
    model_config: RidgeModelConfig = DEFAULT_RIDGE_MODEL_CONFIG,
    seasonal_naive_panel: pd.DataFrame | None = None,
) -> ModelWindowEvaluation:
```

Inside `evaluate_model_window`, after `predictions = add_ridge_predictions(...)`, add:

```python
    if seasonal_naive_panel is None:
        seasonal_naive_panel = derive_seasonal_naive_panel(supervised_frame)
    predictions = add_seasonal_naive_predictions(predictions, seasonal_naive_panel)
```

Then compute Seasonal Naive metrics after Ridge horizon metrics:

```python
    seasonal_naive_overall_metrics = calculate_regression_metrics(
        predictions["target"],
        predictions["seasonal_naive_prediction"],
    )
    seasonal_naive_horizon_metrics = summarize_by_group(
        predictions,
        group_columns=("forecast_horizon",),
        actual_column="target",
        prediction_column="seasonal_naive_prediction",
    )
    model_comparison = _comparison_metrics(
        overall_metrics,
        seasonal_naive_overall_metrics,
    )
```

Add the new fields to the returned `ModelWindowEvaluation`.

In `evaluate_rolling_origin_ridge`, derive the panel once:

```python
    seasonal_naive_panel = derive_seasonal_naive_panel(supervised_frame)
```

Pass it into each `evaluate_model_window` call:

```python
evaluate_model_window(
    supervised_frame,
    window,
    model_config=model_config,
    seasonal_naive_panel=seasonal_naive_panel,
)
```

- [ ] **Step 4: Run evaluation tests**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_evaluation.py -q
```

Expected result: all evaluation tests pass.

- [ ] **Step 5: Commit evaluation metrics**

Run:

```powershell
git add src/urbanflow/modeling/evaluation.py tests/unit/modeling/test_evaluation.py
git commit -m "feat: evaluate seasonal naive beside ridge"
```

## Task 3: Add comparison fields to CLI JSON

**Files:**

- Modify: `tests/unit/modeling/test_modeling_cli.py`
- Modify: `src/urbanflow/modeling/cli.py`

- [ ] **Step 1: Write failing CLI JSON tests**

In `tests/unit/modeling/test_modeling_cli.py`, add this import:

```python
from types import SimpleNamespace
```

In `tests/unit/modeling/test_modeling_cli.py`, add these assertions to `test_ridge_evaluation_cli_returns_json_summary` after the existing final-test metric assertions:

```python
    final_test = payload["final_test"]
    assert final_test["seasonal_naive_overall"]["row_count"] == 672
    assert final_test["seasonal_naive_horizon_metrics"][0]["row_count"] == 672
    assert_finite_metric(final_test["seasonal_naive_overall"]["mae"])
    assert_finite_metric(final_test["seasonal_naive_overall"]["rmse"])
    assert_finite_metric(final_test["seasonal_naive_overall"]["wape"])
    assert final_test["model_comparison"]["ridge_wape"] == final_test["overall"]["wape"]
    assert (
        final_test["model_comparison"]["seasonal_naive_wape"]
        == final_test["seasonal_naive_overall"]["wape"]
    )
    assert_finite_metric(final_test["model_comparison"]["relative_wape_improvement"])
```

Add this test after invalid-options test:

```python
def test_ridge_evaluation_cli_returns_two_for_conflicting_seasonal_naive_panel(
    tmp_path,
    capsys,
) -> None:
    frame = supervised_rows()
    duplicate = frame.iloc[[0]].copy()
    duplicate["target"] = duplicate["target"] + 1.0
    frame = pd.concat([frame, duplicate], ignore_index=True)
    path = tmp_path / "conflicting_supervised_rows.csv"
    frame.to_csv(path, index=False)

    exit_code = main([str(path), "--validation-months", "1"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "conflicting target values for duplicate location_id and target_observed_at" in captured.err
```

Add this test after the conflicting-panel test:

```python
def test_ridge_evaluation_cli_returns_two_when_seasonal_naive_is_unavailable(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    path = write_supervised_csv(tmp_path)
    window = SimpleNamespace(
        name="validation_2025-01",
        start=pd.Timestamp("2025-01-01", tz="Australia/Melbourne"),
        end=pd.Timestamp("2025-02-01", tz="Australia/Melbourne"),
        train_end=pd.Timestamp("2025-01-01", tz="Australia/Melbourne"),
    )
    ridge_metrics = SimpleNamespace(row_count=1, mae=1.0, rmse=1.0, wape=0.1)
    seasonal_naive_metrics = SimpleNamespace(row_count=0, mae=None, rmse=None, wape=None)
    fake_window = SimpleNamespace(
        window=window,
        predictions=pd.DataFrame(),
        overall_metrics=ridge_metrics,
        horizon_metrics=pd.DataFrame(
            [{"forecast_horizon": 1, "row_count": 1, "mae": 1.0, "rmse": 1.0, "wape": 0.1}]
        ),
        model=SimpleNamespace(training_row_count=1),
        seasonal_naive_overall_metrics=seasonal_naive_metrics,
        seasonal_naive_horizon_metrics=pd.DataFrame(
            [{"forecast_horizon": 1, "row_count": 0, "mae": None, "rmse": None, "wape": None}]
        ),
        model_comparison=SimpleNamespace(
            ridge_wape=0.1,
            seasonal_naive_wape=None,
            relative_wape_improvement=None,
        ),
    )
    fake_evaluation = SimpleNamespace(validation_windows=(fake_window,), final_test=fake_window)

    monkeypatch.setattr(
        "urbanflow.modeling.cli.evaluate_rolling_origin_ridge",
        lambda *args, **kwargs: fake_evaluation,
    )

    exit_code = main([str(path), "--validation-months", "1"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Seasonal Naive baseline unavailable for all evaluation windows" in captured.err
```

- [ ] **Step 2: Run the focused CLI tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_cli.py::test_ridge_evaluation_cli_returns_json_summary tests/unit/modeling/test_modeling_cli.py::test_ridge_evaluation_cli_returns_two_for_conflicting_seasonal_naive_panel tests/unit/modeling/test_modeling_cli.py::test_ridge_evaluation_cli_returns_two_when_seasonal_naive_is_unavailable -q
```

Expected result: missing JSON keys, no conflicting-panel CLI error yet, and no unavailable-baseline CLI error yet.

- [ ] **Step 3: Implement JSON summaries and unavailable-baseline check**

In `src/urbanflow/modeling/cli.py`, import the baseline error:

```python
from urbanflow.modeling.baselines import SeasonalNaiveBaselineError
```

Add this helper after `_metrics_summary`:

```python
def _comparison_summary(window_evaluation: ModelWindowEvaluation) -> dict[str, object]:
    comparison = window_evaluation.model_comparison
    return {
        "ridge_wape": comparison.ridge_wape,
        "seasonal_naive_wape": comparison.seasonal_naive_wape,
        "relative_wape_improvement": comparison.relative_wape_improvement,
    }
```

In `_window_summary`, add these fields:

```python
        "seasonal_naive_overall": _metrics_summary(
            evaluation.seasonal_naive_overall_metrics
        ),
        "seasonal_naive_horizon_metrics": _horizon_metric_records(
            evaluation.seasonal_naive_horizon_metrics
        ),
        "model_comparison": _comparison_summary(evaluation),
```

Add this helper after `evaluation_summary`:

```python
def _seasonal_naive_metric_row_count(evaluation: RollingOriginRidgeEvaluation) -> int:
    windows = [*evaluation.validation_windows, evaluation.final_test]
    return sum(window.seasonal_naive_overall_metrics.row_count for window in windows)
```

In `run_ridge_evaluation`, after `evaluation = evaluate_rolling_origin_ridge(...)`, add:

```python
    if _seasonal_naive_metric_row_count(evaluation) == 0:
        raise RidgeEvaluationCliError(
            "Seasonal Naive baseline unavailable for all evaluation windows"
        )
```

In `main`, catch `SeasonalNaiveBaselineError` with the existing expected errors:

```python
    except (
        ModelTrainingError,
        RidgeEvaluationCliError,
        SeasonalNaiveBaselineError,
        SplitConfigError,
    ) as exc:
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_cli.py -q
```

Expected result: all CLI tests pass.

- [ ] **Step 5: Run modeling tests**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling -q
```

Expected result: modeling tests pass, except report example drift may fail after report changes in the next tasks. If it fails only because the checked-in example JSON lacks new fields, proceed to Task 5 before committing all report/example changes together. If any non-report test fails, fix before proceeding.

- [ ] **Step 6: Commit CLI JSON changes when targeted CLI tests are green**

Run:

```powershell
git add src/urbanflow/modeling/cli.py tests/unit/modeling/test_modeling_cli.py
git commit -m "feat: include seasonal naive comparison in evaluation json"
```

## Task 4: Render the Markdown model-comparison table

**Files:**

- Modify: `tests/unit/modeling/test_modeling_reports.py`
- Modify: `src/urbanflow/modeling/reports.py`

- [ ] **Step 1: Extend the report test summary with comparison fields**

In `tests/unit/modeling/test_modeling_reports.py`, update `ridge_summary()` so each validation and final-test window has:

```python
"seasonal_naive_overall": {
    "row_count": 744,
    "mae": 1.4,
    "rmse": 1.9,
    "wape": 0.09,
},
"seasonal_naive_horizon_metrics": [
    {
        "forecast_horizon": 1,
        "row_count": 744,
        "mae": 1.4,
        "rmse": 1.9,
        "wape": 0.09,
    }
],
"model_comparison": {
    "ridge_wape": 0.08123,
    "seasonal_naive_wape": 0.09,
    "relative_wape_improvement": 0.09744444444444444,
},
```

For the final-test window, use:

```python
"seasonal_naive_overall": {
    "row_count": 672,
    "mae": 1.8,
    "rmse": 2.3,
    "wape": 0.1,
},
"seasonal_naive_horizon_metrics": [
    {
        "forecast_horizon": 1,
        "row_count": 672,
        "mae": 1.8,
        "rmse": 2.3,
        "wape": 0.1,
    }
],
"model_comparison": {
    "ridge_wape": 0.07,
    "seasonal_naive_wape": 0.1,
    "relative_wape_improvement": 0.3,
},
```

Add this test after core-section test:

```python
def test_render_ridge_evaluation_report_includes_model_comparison_when_available() -> None:
    markdown = render_ridge_evaluation_report(ridge_summary())

    assert markdown.index("## Final test") < markdown.index("## Model comparison")
    assert markdown.index("## Model comparison") < markdown.index("## Validation windows")
    assert "| Window | Model | Rows | MAE | RMSE | WAPE | Relative WAPE improvement |" in markdown
    assert (
        "| final_test_2025-02 | Ridge | 672 | 1.2000 | 1.7000 | 0.0700 | 30.00% |"
        in markdown
    )
    assert (
        "| final_test_2025-02 | Seasonal Naive | 672 | 1.8000 | 2.3000 | 0.1000 | n/a |"
        in markdown
    )
    assert (
        "| validation_2025-01 | Ridge | 744 | 1.2346 | 1.7543 | 0.0812 | 9.74% |"
        in markdown
    )
```

Add this backward-compatibility test:

```python
def test_render_ridge_evaluation_report_omits_model_comparison_for_older_summaries() -> None:
    summary = deepcopy(ridge_summary())
    final_test = summary["final_test"]
    assert isinstance(final_test, dict)
    final_test.pop("seasonal_naive_overall")
    final_test.pop("seasonal_naive_horizon_metrics")
    final_test.pop("model_comparison")
    validation_windows = summary["validation_windows"]
    assert isinstance(validation_windows, list)
    validation_window = validation_windows[0]
    assert isinstance(validation_window, dict)
    validation_window.pop("seasonal_naive_overall")
    validation_window.pop("seasonal_naive_horizon_metrics")
    validation_window.pop("model_comparison")

    markdown = render_ridge_evaluation_report(summary)

    assert "## Model comparison" not in markdown
    assert "## Validation windows" in markdown
```

- [ ] **Step 2: Run the report tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py::test_render_ridge_evaluation_report_includes_model_comparison_when_available tests/unit/modeling/test_modeling_reports.py::test_render_ridge_evaluation_report_omits_model_comparison_for_older_summaries -q
```

Expected result: the inclusion test fails because the report renderer does not yet emit `## Model comparison`.

- [ ] **Step 3: Implement optional model-comparison rendering**

In `src/urbanflow/modeling/reports.py`, add this helper after `_metric_text`:

```python
def _percent_text(value: Any) -> str:
    metric = _numeric_metric_value(value)
    if metric is None:
        return "n/a"
    return f"{metric * 100:.2f}%"
```

Add these helpers after `_validation_row`:

```python
def _has_model_comparison(window: Mapping[str, Any]) -> bool:
    return "seasonal_naive_overall" in window and "model_comparison" in window


def _model_comparison_row(window: Mapping[str, Any], *, model_name: str) -> str:
    if model_name == "Ridge":
        metrics = _metric_mapping(window, path=str(window["name"]))
        comparison = _required_mapping(
            window["model_comparison"],
            path=f"{window['name']}.model_comparison",
        )
        improvement = _percent_text(comparison.get("relative_wape_improvement"))
    else:
        metrics = _required_mapping(
            window["seasonal_naive_overall"],
            path=f"{window['name']}.seasonal_naive_overall",
        )
        improvement = "n/a"
    return (
        f"| {_cell_text(window['name'])} | {model_name} | "
        f"{_count_text(metrics['row_count'])} | "
        f"{_metric_text(metrics['mae'])} | "
        f"{_metric_text(metrics['rmse'])} | "
        f"{_metric_text(metrics['wape'])} | "
        f"{improvement} |"
    )


def _model_comparison_lines(
    final_test: Mapping[str, Any],
    validation_windows: Sequence[Mapping[str, Any]],
) -> list[str]:
    windows = [final_test, *validation_windows]
    comparable_windows = [window for window in windows if _has_model_comparison(window)]
    if not comparable_windows:
        return []

    lines = [
        "",
        "## Model comparison",
        "",
        "| Window | Model | Rows | MAE | RMSE | WAPE | Relative WAPE improvement |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for window in comparable_windows:
        lines.append(_model_comparison_row(window, model_name="Ridge"))
        lines.append(_model_comparison_row(window, model_name="Seasonal Naive"))
    return lines
```

In `render_ridge_evaluation_report`, split the initial `lines` list so it ends
immediately after the final-test WAPE row:

```python
        f"| WAPE | {_metric_text(final_overall['wape'])} |",
    ]
```

Then insert the model-comparison lines and append the validation table header:

```python
    lines.extend(_model_comparison_lines(final_test, validation_window_mappings))
    lines.extend(
        [
            "",
            "## Validation windows",
            "",
            "| Window | Period | Training rows | Rows | MAE | RMSE | WAPE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
```

- [ ] **Step 4: Run report tests**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -q
```

Expected result: report tests pass except the checked-in example drift test may fail until Task 5 updates the example JSON and Markdown report.

- [ ] **Step 5: Commit report renderer only after checked-in example is updated in Task 5**

Do not commit this task by itself if the checked-in example drift test is failing. Keep the changes staged for Task 5's combined report/example commit.

## Task 5: Update checked-in example and README

**Files:**

- Modify: `docs/examples/modeling/ridge_evaluation_summary.json`
- Modify: `docs/examples/modeling/ridge_evaluation_report.md`
- Modify: `README.md`
- Test: `tests/unit/modeling/test_modeling_reports.py`

- [ ] **Step 1: Add Seasonal Naive fields to the checked-in example summary**

Update `docs/examples/modeling/ridge_evaluation_summary.json` so the validation window includes:

```json
"seasonal_naive_overall": {
  "row_count": 744,
  "mae": 1.42,
  "rmse": 1.98,
  "wape": 0.093
},
"seasonal_naive_horizon_metrics": [
  {
    "forecast_horizon": 1,
    "row_count": 744,
    "mae": 1.42,
    "rmse": 1.98,
    "wape": 0.093
  }
],
"model_comparison": {
  "ridge_wape": 0.0812,
  "seasonal_naive_wape": 0.093,
  "relative_wape_improvement": 0.1268817204301075
}
```

Update the final-test window with:

```json
"seasonal_naive_overall": {
  "row_count": 672,
  "mae": 1.8,
  "rmse": 2.35,
  "wape": 0.095
},
"seasonal_naive_horizon_metrics": [
  {
    "forecast_horizon": 1,
    "row_count": 336,
    "mae": 1.7,
    "rmse": 2.2,
    "wape": 0.09
  },
  {
    "forecast_horizon": 24,
    "row_count": 336,
    "mae": 1.9,
    "rmse": 2.5,
    "wape": 0.1
  }
],
"model_comparison": {
  "ridge_wape": 0.07,
  "seasonal_naive_wape": 0.095,
  "relative_wape_improvement": 0.2631578947368421
}
```

- [ ] **Step 2: Regenerate the checked-in Markdown report**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' scripts/render_ridge_evaluation_report.py docs/examples/modeling/ridge_evaluation_summary.json --output docs/examples/modeling/ridge_evaluation_report.md --force
```

Expected result: stdout contains JSON with output path `docs\examples\modeling\ridge_evaluation_report.md` or the same path with `/`.

- [ ] **Step 3: Update README model-report wording**

In `README.md`, after the sentence about Mermaid comparison charts, add:

```markdown
The same report also includes a Ridge versus Seasonal Naive comparison table so
the trainable baseline can be interpreted against a one-week-prior baseline.
```

- [ ] **Step 4: Run report tests**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/modeling/test_modeling_reports.py -q
```

Expected result: all report tests pass, including the checked-in example drift test.

- [ ] **Step 5: Commit report and example changes**

Run:

```powershell
git add src/urbanflow/modeling/reports.py tests/unit/modeling/test_modeling_reports.py docs/examples/modeling/ridge_evaluation_summary.json docs/examples/modeling/ridge_evaluation_report.md README.md
git commit -m "feat: show seasonal naive comparison in ridge report"
```

## Task 6: Final verification before merge

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

Expected result: command exits `0`.

- [ ] **Step 3: Run the full pytest suite**

Run:

```powershell
$env:PYTHONPATH='src'; & 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest
```

Expected result: all tests pass. The current suite has 147 tests before this implementation; the count should increase after the new tests land.

- [ ] **Step 4: Inspect final state**

Run:

```powershell
git status --short
git diff --check
git log --oneline -8
```

Expected result:

- `git status --short` shows no uncommitted files.
- `git diff --check` prints no whitespace errors.
- recent commits include:
  - `feat: derive seasonal naive panel from supervised rows`
  - `feat: evaluate seasonal naive beside ridge`
  - `feat: include seasonal naive comparison in evaluation json`
  - `feat: show seasonal naive comparison in ridge report`

## Implementation self-review checklist

Before merging to `main`, verify these requirements against the diff:

- Seasonal Naive panel is derived from `location_id`, `target_observed_at`, and `target`.
- Matching duplicate target rows are deduplicated.
- Conflicting duplicate target rows return an expected user error.
- Ridge and Seasonal Naive metrics use identical evaluation windows.
- Seasonal Naive missing history drops only Seasonal Naive metric rows, not Ridge rows.
- CLI returns code `2` when Seasonal Naive is unavailable across all evaluation windows.
- JSON contains `seasonal_naive_overall`, `seasonal_naive_horizon_metrics`, and `model_comparison`.
- `relative_wape_improvement` follows `(seasonal_naive_wape - ridge_wape) / seasonal_naive_wape`.
- Markdown renders `## Model comparison` only when comparison fields are present.
- Older Ridge-only summaries still render without `## Model comparison`.
- Checked-in example JSON and Markdown report stay in exact renderer sync.
- README explains the Ridge versus Seasonal Naive comparison without implying unreal model performance.
