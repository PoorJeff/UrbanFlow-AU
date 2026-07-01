# Ridge Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first leakage-safe trainable Ridge Regression baseline on top of the existing supervised multi-horizon feature rows.

**Architecture:** Add a narrow `urbanflow.modeling.feature_matrix` contract, a scikit-learn Ridge wrapper in `urbanflow.modeling.ridge`, and windowed evaluation helpers in `urbanflow.modeling.evaluation`. Keep this slice DataFrame-first, deterministic, and independent of PostgreSQL, MLflow, LightGBM, dashboard code, or model artifact persistence.

**Tech Stack:** Python 3.11+, pandas, scikit-learn, pytest, Ruff, existing `urbanflow.features` and `urbanflow.modeling` utilities.

---

## Source spec

Implement:

`docs/superpowers/specs/2026-07-01-ridge-baseline-design.md`

## Worktree and execution note

Create an isolated worktree before executing code tasks:

```powershell
git worktree add '.worktrees/ridge-baseline-implementation' -b codex/ridge-baseline-implementation
cd '.worktrees/ridge-baseline-implementation'
$env:PYTHONPATH='src'
```

Use the existing project virtual environment when available:

```powershell
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pip install -e ".[dev]"
```

If using a shell where `python` already resolves to the project virtual environment, the shorter commands in the tasks are equivalent.

## File structure

- Modify `pyproject.toml`
  - Add `scikit-learn>=1.5,<2` to project dependencies.
- Create `src/urbanflow/modeling/feature_matrix.py`
  - Own the Ridge-safe feature whitelist, target filtering, and missing-column checks.
- Create `src/urbanflow/modeling/ridge.py`
  - Own Ridge configuration, fitted model wrapper, training, and prediction.
- Create `src/urbanflow/modeling/evaluation.py`
  - Own windowed Ridge evaluation and rolling-origin Ridge evaluation.
- Modify `src/urbanflow/modeling/__init__.py`
  - Export the new public modeling helpers.
- Create `tests/unit/modeling/test_feature_matrix.py`
  - Test the leakage-safe feature contract.
- Create `tests/unit/modeling/test_ridge.py`
  - Test Ridge fitting, prediction, and unknown location handling.
- Create `tests/unit/modeling/test_evaluation.py`
  - Test time-window filtering and metric output.
- Modify `README.md`
  - Add a short Ridge baseline note after the leakage-safe modeling features section.

## Task 1: Dependency and leakage-safe feature matrix contract

**Files:**
- Modify: `pyproject.toml`
- Create: `src/urbanflow/modeling/feature_matrix.py`
- Modify: `src/urbanflow/modeling/__init__.py`
- Create: `tests/unit/modeling/test_feature_matrix.py`

- [ ] **Step 1: Declare scikit-learn dependency**

Modify `pyproject.toml` project dependencies so the dependency block includes:

```toml
dependencies = [
    "alembic>=1.13,<2",
    "httpx>=0.28,<1",
    "pandas>=2.1,<4",
    "pandera[pandas]>=0.24,<1",
    "prefect>=3,<4",
    "psycopg[binary]>=3.2,<4",
    "scikit-learn>=1.5,<2",
    "SQLAlchemy>=2.0,<3",
    "tenacity>=9,<10",
]
```

- [ ] **Step 2: Install updated project dependencies**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pip install -e ".[dev]"
```

Expected: install completes and `python -m pip show scikit-learn` prints a version in the `>=1.5,<2` range.

- [ ] **Step 3: Write failing feature matrix tests**

Create `tests/unit/modeling/test_feature_matrix.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from urbanflow.modeling.feature_matrix import (
    DEFAULT_RIDGE_FEATURE_SPEC,
    ModelFeatureSpec,
    ModelTrainingError,
    select_model_features,
    select_training_rows,
)


def supervised_rows() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 00:00", periods=3, freq="h", tz="Australia/Melbourne")
    return pd.DataFrame(
        {
            "location_id": [101, 101, 102],
            "forecast_origin_at": timestamps,
            "forecast_horizon": [1, 2, 1],
            "target_observed_at": timestamps + pd.Timedelta(hours=1),
            "target": [100.0, 120.0, None],
            "target_missing": [False, False, True],
            "pedestrian_count": [95.0, 110.0, 130.0],
            "pedestrian_count_missing": [False, False, False],
            "lag_1": [95.0, 110.0, 130.0],
            "lag_24": [80.0, 90.0, 100.0],
            "lag_168": [70.0, 85.0, 95.0],
            "rolling_24_mean": [90.0, 100.0, 110.0],
            "rolling_24_std": [5.0, 6.0, 7.0],
            "rolling_168_mean": [88.0, 98.0, 108.0],
            "rolling_168_std": [8.0, 9.0, 10.0],
            "hour": [1, 2, 3],
            "weekday": [2, 2, 2],
            "month": [1, 1, 1],
            "is_weekend": [False, False, False],
            "is_public_holiday": [False, False, False],
            "hour_sin": [0.1, 0.2, 0.3],
            "hour_cos": [0.9, 0.8, 0.7],
            "weekday_sin": [0.4, 0.4, 0.4],
            "weekday_cos": [0.5, 0.5, 0.5],
            "temperature": [20.0, 21.0, None],
            "temperature_missing": [False, False, True],
            "rainfall": [0.0, 0.2, 0.0],
            "rainfall_missing": [False, False, False],
            "wind_speed": [12.0, None, 15.0],
            "wind_speed_missing": [False, True, False],
            "seasonal_naive_prediction": [98.0, 115.0, 125.0],
            "seasonal_naive_missing": [False, False, False],
            "seasonal_naive_observed_at": timestamps - pd.Timedelta(hours=168),
            "ridge_prediction": [99.0, 119.0, 129.0],
        }
    )


def test_default_ridge_feature_spec_whitelists_safe_features() -> None:
    spec = DEFAULT_RIDGE_FEATURE_SPEC

    assert spec.categorical_columns == ("location_id",)
    assert "forecast_horizon" in spec.numeric_columns
    assert "lag_168" in spec.numeric_columns
    assert "is_public_holiday" in spec.numeric_columns

    excluded_columns = {
        "target",
        "target_missing",
        "target_observed_at",
        "forecast_origin_at",
        "seasonal_naive_prediction",
        "seasonal_naive_missing",
        "seasonal_naive_observed_at",
        "ridge_prediction",
    }
    assert excluded_columns.isdisjoint(spec.feature_columns)


def test_select_training_rows_drops_missing_targets() -> None:
    rows = select_training_rows(supervised_rows())

    assert len(rows) == 2
    assert rows["target"].tolist() == [100.0, 120.0]


def test_select_model_features_returns_ordered_features_and_target() -> None:
    features, target = select_model_features(supervised_rows())

    assert tuple(features.columns) == DEFAULT_RIDGE_FEATURE_SPEC.feature_columns
    assert target.tolist() == [100.0, 120.0]
    assert "target_observed_at" not in features.columns
    assert "seasonal_naive_prediction" not in features.columns


def test_select_model_features_rejects_missing_required_columns() -> None:
    frame = supervised_rows().drop(columns=["lag_24"])

    with pytest.raises(ModelTrainingError, match="missing required columns: lag_24"):
        select_model_features(frame)


def test_custom_feature_spec_is_supported() -> None:
    spec = ModelFeatureSpec(
        numeric_columns=("forecast_horizon", "lag_1"),
        categorical_columns=("location_id",),
    )

    features, target = select_model_features(supervised_rows(), feature_spec=spec)

    assert tuple(features.columns) == ("forecast_horizon", "lag_1", "location_id")
    assert target.tolist() == [100.0, 120.0]
```

- [ ] **Step 4: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/modeling/test_feature_matrix.py -v
```

Expected: FAIL because `urbanflow.modeling.feature_matrix` does not exist.

- [ ] **Step 5: Implement feature matrix contract**

Create `src/urbanflow/modeling/feature_matrix.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ModelTrainingError(ValueError):
    """Raised when model training or prediction inputs are invalid."""


@dataclass(frozen=True)
class ModelFeatureSpec:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...] = ("location_id",)
    target_column: str = "target"
    target_missing_column: str = "target_missing"

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.categorical_columns

    @property
    def required_training_columns(self) -> tuple[str, ...]:
        return self.feature_columns + (self.target_column, self.target_missing_column)


DEFAULT_RIDGE_FEATURE_SPEC = ModelFeatureSpec(
    numeric_columns=(
        "forecast_horizon",
        "pedestrian_count",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_24_mean",
        "rolling_24_std",
        "rolling_168_mean",
        "rolling_168_std",
        "hour",
        "weekday",
        "month",
        "is_weekend",
        "is_public_holiday",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "temperature",
        "temperature_missing",
        "rainfall",
        "rainfall_missing",
        "wind_speed",
        "wind_speed_missing",
        "pedestrian_count_missing",
    ),
    categorical_columns=("location_id",),
)


def _missing_columns(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(column for column in required_columns if column not in frame.columns)


def validate_model_feature_columns(
    frame: pd.DataFrame,
    *,
    feature_spec: ModelFeatureSpec = DEFAULT_RIDGE_FEATURE_SPEC,
    require_target: bool = True,
) -> None:
    required_columns = (
        feature_spec.required_training_columns if require_target else feature_spec.feature_columns
    )
    missing = _missing_columns(frame, required_columns)
    if missing:
        raise ModelTrainingError(f"missing required columns: {', '.join(missing)}")


def select_training_rows(
    frame: pd.DataFrame,
    *,
    feature_spec: ModelFeatureSpec = DEFAULT_RIDGE_FEATURE_SPEC,
) -> pd.DataFrame:
    validate_model_feature_columns(frame, feature_spec=feature_spec, require_target=True)
    target_missing = frame[feature_spec.target_column].isna() | frame[
        feature_spec.target_missing_column
    ].fillna(True).astype(bool)
    rows = frame.loc[~target_missing].copy()
    if rows.empty:
        raise ModelTrainingError("no training rows remain after filtering missing targets")
    return rows


def select_model_features(
    frame: pd.DataFrame,
    *,
    feature_spec: ModelFeatureSpec = DEFAULT_RIDGE_FEATURE_SPEC,
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    if require_target:
        rows = select_training_rows(frame, feature_spec=feature_spec)
        target = rows[feature_spec.target_column].astype(float)
    else:
        validate_model_feature_columns(frame, feature_spec=feature_spec, require_target=False)
        rows = frame.copy()
        target = None
    return rows[list(feature_spec.feature_columns)].copy(), target
```

- [ ] **Step 6: Export feature matrix helpers**

Modify `src/urbanflow/modeling/__init__.py`:

```python
from urbanflow.modeling.baselines import add_seasonal_naive_predictions
from urbanflow.modeling.feature_matrix import (
    DEFAULT_RIDGE_FEATURE_SPEC,
    ModelFeatureSpec,
    ModelTrainingError,
    select_model_features,
    select_training_rows,
    validate_model_feature_columns,
)
from urbanflow.modeling.metrics import (
    RegressionMetrics,
    calculate_regression_metrics,
    peak_top_decile_mae,
    summarize_by_group,
)
from urbanflow.modeling.splits import (
    EvaluationWindow,
    RollingOriginSplits,
    SplitConfigError,
    build_rolling_origin_splits,
    complete_months,
)

__all__ = [
    "DEFAULT_RIDGE_FEATURE_SPEC",
    "EvaluationWindow",
    "ModelFeatureSpec",
    "ModelTrainingError",
    "RegressionMetrics",
    "RollingOriginSplits",
    "SplitConfigError",
    "add_seasonal_naive_predictions",
    "build_rolling_origin_splits",
    "calculate_regression_metrics",
    "complete_months",
    "peak_top_decile_mae",
    "select_model_features",
    "select_training_rows",
    "summarize_by_group",
    "validate_model_feature_columns",
]
```

- [ ] **Step 7: Run focused tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/modeling/test_feature_matrix.py -v
```

Expected: PASS.

- [ ] **Step 8: Run targeted quality checks**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check src/urbanflow/modeling tests/unit/modeling --no-cache
python -m ruff format --check src/urbanflow/modeling tests/unit/modeling
```

Expected: both commands pass.

- [ ] **Step 9: Commit feature matrix task**

Run:

```powershell
git add pyproject.toml src/urbanflow/modeling/__init__.py src/urbanflow/modeling/feature_matrix.py tests/unit/modeling/test_feature_matrix.py
git commit -m "feat: add ridge feature matrix contract"
```

Expected: one commit containing dependency declaration, feature matrix contract, exports, and tests.

## Task 2: Ridge training and prediction wrapper

**Files:**
- Create: `src/urbanflow/modeling/ridge.py`
- Modify: `src/urbanflow/modeling/__init__.py`
- Create: `tests/unit/modeling/test_ridge.py`

- [ ] **Step 1: Write failing Ridge tests**

Create `tests/unit/modeling/test_ridge.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urbanflow.modeling.feature_matrix import DEFAULT_RIDGE_FEATURE_SPEC, ModelTrainingError
from urbanflow.modeling.ridge import RidgeModelConfig, add_ridge_predictions, fit_ridge_model


def supervised_rows(location_ids: tuple[int, ...] = (101, 101, 102, 102)) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01 00:00",
        periods=len(location_ids),
        freq="h",
        tz="Australia/Melbourne",
    )
    base = pd.DataFrame(
        {
            "location_id": list(location_ids),
            "forecast_origin_at": timestamps,
            "forecast_horizon": [1, 2, 1, 2][: len(location_ids)],
            "target_observed_at": timestamps + pd.Timedelta(hours=1),
            "target": [100.0, 120.0, 130.0, 150.0][: len(location_ids)],
            "target_missing": [False] * len(location_ids),
            "pedestrian_count": [95.0, 110.0, 125.0, 140.0][: len(location_ids)],
            "pedestrian_count_missing": [False] * len(location_ids),
            "lag_1": [95.0, 110.0, 125.0, 140.0][: len(location_ids)],
            "lag_24": [80.0, 90.0, 100.0, 115.0][: len(location_ids)],
            "lag_168": [70.0, 85.0, 95.0, 105.0][: len(location_ids)],
            "rolling_24_mean": [90.0, 100.0, 115.0, 130.0][: len(location_ids)],
            "rolling_24_std": [5.0, 6.0, 7.0, 8.0][: len(location_ids)],
            "rolling_168_mean": [88.0, 98.0, 108.0, 118.0][: len(location_ids)],
            "rolling_168_std": [8.0, 9.0, 10.0, 11.0][: len(location_ids)],
            "hour": [1, 2, 3, 4][: len(location_ids)],
            "weekday": [2] * len(location_ids),
            "month": [1] * len(location_ids),
            "is_weekend": [False] * len(location_ids),
            "is_public_holiday": [False] * len(location_ids),
            "hour_sin": [0.1, 0.2, 0.3, 0.4][: len(location_ids)],
            "hour_cos": [0.9, 0.8, 0.7, 0.6][: len(location_ids)],
            "weekday_sin": [0.4] * len(location_ids),
            "weekday_cos": [0.5] * len(location_ids),
            "temperature": [20.0, 21.0, None, 23.0][: len(location_ids)],
            "temperature_missing": [False, False, True, False][: len(location_ids)],
            "rainfall": [0.0, 0.2, 0.0, 0.1][: len(location_ids)],
            "rainfall_missing": [False] * len(location_ids),
            "wind_speed": [12.0, None, 15.0, 13.0][: len(location_ids)],
            "wind_speed_missing": [False, True, False, False][: len(location_ids)],
        }
    )
    return base


def test_fit_ridge_model_records_metadata() -> None:
    model = fit_ridge_model(supervised_rows(), config=RidgeModelConfig(alpha=0.5))

    assert model.training_row_count == 4
    assert model.feature_columns == DEFAULT_RIDGE_FEATURE_SPEC.feature_columns
    assert model.config.alpha == 0.5


def test_add_ridge_predictions_returns_copy_with_finite_predictions() -> None:
    frame = supervised_rows()
    model = fit_ridge_model(frame)

    result = add_ridge_predictions(frame, model)

    assert result is not frame
    assert "ridge_prediction" in result.columns
    assert len(result) == len(frame)
    assert np.isfinite(result["ridge_prediction"]).all()


def test_add_ridge_predictions_handles_unknown_location_id() -> None:
    model = fit_ridge_model(supervised_rows())
    prediction_frame = supervised_rows(location_ids=(999, 999))

    result = add_ridge_predictions(prediction_frame, model)

    assert len(result) == 2
    assert np.isfinite(result["ridge_prediction"]).all()


def test_fit_ridge_model_rejects_empty_training_rows() -> None:
    frame = supervised_rows()
    frame["target"] = None
    frame["target_missing"] = True

    with pytest.raises(ModelTrainingError, match="no training rows"):
        fit_ridge_model(frame)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/modeling/test_ridge.py -v
```

Expected: FAIL because `urbanflow.modeling.ridge` does not exist.

- [ ] **Step 3: Implement Ridge wrapper**

Create `src/urbanflow/modeling/ridge.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from urbanflow.modeling.feature_matrix import (
    DEFAULT_RIDGE_FEATURE_SPEC,
    ModelFeatureSpec,
    ModelTrainingError,
    select_model_features,
)


@dataclass(frozen=True)
class RidgeModelConfig:
    alpha: float = 1.0
    feature_spec: ModelFeatureSpec = DEFAULT_RIDGE_FEATURE_SPEC
    prediction_column: str = "ridge_prediction"


@dataclass(frozen=True)
class FittedRidgeModel:
    pipeline: Pipeline
    config: RidgeModelConfig
    feature_columns: tuple[str, ...]
    training_row_count: int

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features, _ = select_model_features(
            frame,
            feature_spec=self.config.feature_spec,
            require_target=False,
        )
        predictions = self.pipeline.predict(features)
        return np.asarray(predictions, dtype=float)


def _build_ridge_pipeline(config: RidgeModelConfig) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(config.feature_spec.numeric_columns)),
            ("categorical", categorical_pipeline, list(config.feature_spec.categorical_columns)),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("features", preprocessor),
            ("model", Ridge(alpha=config.alpha)),
        ]
    )


def fit_ridge_model(
    train_frame: pd.DataFrame,
    *,
    config: RidgeModelConfig = RidgeModelConfig(),
) -> FittedRidgeModel:
    features, target = select_model_features(train_frame, feature_spec=config.feature_spec)
    if target is None:
        raise ModelTrainingError("target is required for Ridge training")

    pipeline = _build_ridge_pipeline(config)
    pipeline.fit(features, target)
    return FittedRidgeModel(
        pipeline=pipeline,
        config=config,
        feature_columns=config.feature_spec.feature_columns,
        training_row_count=len(features),
    )


def add_ridge_predictions(frame: pd.DataFrame, fitted_model: FittedRidgeModel) -> pd.DataFrame:
    result = frame.copy()
    predictions = fitted_model.predict(result)
    result[fitted_model.config.prediction_column] = predictions
    return result
```

- [ ] **Step 4: Export Ridge helpers**

Modify `src/urbanflow/modeling/__init__.py` to add these imports:

```python
from urbanflow.modeling.ridge import (
    FittedRidgeModel,
    RidgeModelConfig,
    add_ridge_predictions,
    fit_ridge_model,
)
```

Add these names to `__all__`:

```python
    "FittedRidgeModel",
    "RidgeModelConfig",
    "add_ridge_predictions",
    "fit_ridge_model",
```

- [ ] **Step 5: Run focused Ridge tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/modeling/test_feature_matrix.py tests/unit/modeling/test_ridge.py -v
```

Expected: PASS.

- [ ] **Step 6: Run targeted quality checks**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check src/urbanflow/modeling tests/unit/modeling --no-cache
python -m ruff format --check src/urbanflow/modeling tests/unit/modeling
```

Expected: both commands pass.

- [ ] **Step 7: Commit Ridge wrapper task**

Run:

```powershell
git add src/urbanflow/modeling/__init__.py src/urbanflow/modeling/ridge.py tests/unit/modeling/test_ridge.py
git commit -m "feat: add ridge training baseline"
```

Expected: one commit containing Ridge training, prediction, exports, and tests.

## Task 3: Rolling-origin Ridge evaluation helpers

**Files:**
- Create: `src/urbanflow/modeling/evaluation.py`
- Modify: `src/urbanflow/modeling/__init__.py`
- Create: `tests/unit/modeling/test_evaluation.py`

- [ ] **Step 1: Write failing evaluation tests**

Create `tests/unit/modeling/test_evaluation.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from urbanflow.modeling.evaluation import (
    RollingOriginRidgeEvaluation,
    evaluate_model_window,
    evaluate_rolling_origin_ridge,
)
from urbanflow.modeling.feature_matrix import ModelTrainingError
from urbanflow.modeling.splits import EvaluationWindow, RollingOriginSplits


def supervised_rows() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 00:00", periods=8, freq="h", tz="Australia/Melbourne")
    values = [80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    return pd.DataFrame(
        {
            "location_id": [101, 101, 101, 101, 102, 102, 102, 102],
            "forecast_origin_at": timestamps - pd.Timedelta(hours=1),
            "forecast_horizon": [1, 2, 1, 2, 1, 2, 1, 2],
            "target_observed_at": timestamps,
            "target": values,
            "target_missing": [False] * 8,
            "pedestrian_count": [value - 5.0 for value in values],
            "pedestrian_count_missing": [False] * 8,
            "lag_1": [value - 5.0 for value in values],
            "lag_24": [value - 10.0 for value in values],
            "lag_168": [value - 20.0 for value in values],
            "rolling_24_mean": [value - 7.0 for value in values],
            "rolling_24_std": [3.0] * 8,
            "rolling_168_mean": [value - 15.0 for value in values],
            "rolling_168_std": [6.0] * 8,
            "hour": list(range(8)),
            "weekday": [2] * 8,
            "month": [1] * 8,
            "is_weekend": [False] * 8,
            "is_public_holiday": [False] * 8,
            "hour_sin": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "hour_cos": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            "weekday_sin": [0.4] * 8,
            "weekday_cos": [0.5] * 8,
            "temperature": [20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5],
            "temperature_missing": [False] * 8,
            "rainfall": [0.0] * 8,
            "rainfall_missing": [False] * 8,
            "wind_speed": [12.0, 11.0, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5],
            "wind_speed_missing": [False] * 8,
        }
    )


def evaluation_window() -> EvaluationWindow:
    return EvaluationWindow(
        name="validation_2025_01_01_04",
        start=pd.Timestamp("2025-01-01 04:00", tz="Australia/Melbourne"),
        end=pd.Timestamp("2025-01-01 08:00", tz="Australia/Melbourne"),
        train_end=pd.Timestamp("2025-01-01 04:00", tz="Australia/Melbourne"),
    )


def test_evaluate_model_window_filters_train_and_evaluation_rows() -> None:
    result = evaluate_model_window(supervised_rows(), evaluation_window())

    assert result.window.name == "validation_2025_01_01_04"
    assert result.model.training_row_count == 4
    assert len(result.predictions) == 4
    assert result.predictions["target_observed_at"].min() >= evaluation_window().start
    assert result.predictions["target_observed_at"].max() < evaluation_window().end
    assert "ridge_prediction" in result.predictions.columns
    assert result.overall_metrics.row_count == 4


def test_evaluate_model_window_returns_per_horizon_metrics() -> None:
    result = evaluate_model_window(supervised_rows(), evaluation_window())

    assert set(result.horizon_metrics["forecast_horizon"]) == {1, 2}
    assert set(result.horizon_metrics.columns) == {
        "forecast_horizon",
        "row_count",
        "mae",
        "rmse",
        "wape",
    }


def test_evaluate_model_window_rejects_empty_evaluation_window() -> None:
    window = EvaluationWindow(
        name="empty",
        start=pd.Timestamp("2025-01-02 00:00", tz="Australia/Melbourne"),
        end=pd.Timestamp("2025-01-02 01:00", tz="Australia/Melbourne"),
        train_end=pd.Timestamp("2025-01-01 04:00", tz="Australia/Melbourne"),
    )

    with pytest.raises(ModelTrainingError, match="no evaluation rows"):
        evaluate_model_window(supervised_rows(), window)


def test_evaluate_rolling_origin_ridge_evaluates_validation_and_final_windows() -> None:
    splits = RollingOriginSplits(
        validation_windows=(evaluation_window(),),
        final_test=EvaluationWindow(
            name="final_test_2025_01_01_06",
            start=pd.Timestamp("2025-01-01 06:00", tz="Australia/Melbourne"),
            end=pd.Timestamp("2025-01-01 08:00", tz="Australia/Melbourne"),
            train_end=pd.Timestamp("2025-01-01 06:00", tz="Australia/Melbourne"),
        ),
    )

    result = evaluate_rolling_origin_ridge(supervised_rows(), splits)

    assert isinstance(result, RollingOriginRidgeEvaluation)
    assert len(result.validation_windows) == 1
    assert result.final_test.window.name == "final_test_2025_01_01_06"
    assert len(result.final_test.predictions) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/modeling/test_evaluation.py -v
```

Expected: FAIL because `urbanflow.modeling.evaluation` does not exist.

- [ ] **Step 3: Implement evaluation helpers**

Create `src/urbanflow/modeling/evaluation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from urbanflow.modeling.feature_matrix import ModelTrainingError
from urbanflow.modeling.metrics import RegressionMetrics, calculate_regression_metrics, summarize_by_group
from urbanflow.modeling.ridge import (
    FittedRidgeModel,
    RidgeModelConfig,
    add_ridge_predictions,
    fit_ridge_model,
)
from urbanflow.modeling.splits import EvaluationWindow, RollingOriginSplits


@dataclass(frozen=True)
class ModelWindowEvaluation:
    window: EvaluationWindow
    predictions: pd.DataFrame
    overall_metrics: RegressionMetrics
    horizon_metrics: pd.DataFrame
    model: FittedRidgeModel


@dataclass(frozen=True)
class RollingOriginRidgeEvaluation:
    validation_windows: tuple[ModelWindowEvaluation, ...]
    final_test: ModelWindowEvaluation


def _training_rows_for_window(frame: pd.DataFrame, window: EvaluationWindow) -> pd.DataFrame:
    rows = frame.loc[frame["target_observed_at"] < window.train_end].copy()
    if rows.empty:
        raise ModelTrainingError(f"no training rows before {window.train_end}")
    return rows


def _evaluation_rows_for_window(frame: pd.DataFrame, window: EvaluationWindow) -> pd.DataFrame:
    rows = frame.loc[
        (frame["target_observed_at"] >= window.start) & (frame["target_observed_at"] < window.end)
    ].copy()
    if rows.empty:
        raise ModelTrainingError(f"no evaluation rows for window {window.name}")
    return rows


def evaluate_model_window(
    supervised_frame: pd.DataFrame,
    window: EvaluationWindow,
    *,
    model_config: RidgeModelConfig = RidgeModelConfig(),
) -> ModelWindowEvaluation:
    training_rows = _training_rows_for_window(supervised_frame, window)
    evaluation_rows = _evaluation_rows_for_window(supervised_frame, window)
    model = fit_ridge_model(training_rows, config=model_config)
    predictions = add_ridge_predictions(evaluation_rows, model)
    overall_metrics = calculate_regression_metrics(
        predictions["target"],
        predictions[model_config.prediction_column],
    )
    horizon_metrics = summarize_by_group(
        predictions,
        group_columns=("forecast_horizon",),
        actual_column="target",
        prediction_column=model_config.prediction_column,
    )
    return ModelWindowEvaluation(
        window=window,
        predictions=predictions,
        overall_metrics=overall_metrics,
        horizon_metrics=horizon_metrics,
        model=model,
    )


def evaluate_rolling_origin_ridge(
    supervised_frame: pd.DataFrame,
    splits: RollingOriginSplits,
    *,
    model_config: RidgeModelConfig = RidgeModelConfig(),
) -> RollingOriginRidgeEvaluation:
    validation_windows = tuple(
        evaluate_model_window(supervised_frame, window, model_config=model_config)
        for window in splits.validation_windows
    )
    final_test = evaluate_model_window(
        supervised_frame,
        splits.final_test,
        model_config=model_config,
    )
    return RollingOriginRidgeEvaluation(
        validation_windows=validation_windows,
        final_test=final_test,
    )
```

- [ ] **Step 4: Export evaluation helpers**

Modify `src/urbanflow/modeling/__init__.py` to add these imports:

```python
from urbanflow.modeling.evaluation import (
    ModelWindowEvaluation,
    RollingOriginRidgeEvaluation,
    evaluate_model_window,
    evaluate_rolling_origin_ridge,
)
```

Add these names to `__all__`:

```python
    "ModelWindowEvaluation",
    "RollingOriginRidgeEvaluation",
    "evaluate_model_window",
    "evaluate_rolling_origin_ridge",
```

- [ ] **Step 5: Run focused evaluation tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/modeling/test_feature_matrix.py tests/unit/modeling/test_ridge.py tests/unit/modeling/test_evaluation.py -v
```

Expected: PASS.

- [ ] **Step 6: Run targeted quality checks**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check src/urbanflow/modeling tests/unit/modeling --no-cache
python -m ruff format --check src/urbanflow/modeling tests/unit/modeling
```

Expected: both commands pass.

- [ ] **Step 7: Commit evaluation task**

Run:

```powershell
git add src/urbanflow/modeling/__init__.py src/urbanflow/modeling/evaluation.py tests/unit/modeling/test_evaluation.py
git commit -m "feat: add ridge rolling-origin evaluation"
```

Expected: one commit containing evaluation helpers, exports, and tests.

## Task 4: README note, full verification, merge, push, and cleanup

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a short README note for Ridge baseline**

In `README.md`, after the existing `## Build leakage-safe modeling features` section, add:

```markdown
## Train a local Ridge baseline

The first trainable model slice fits a leakage-safe Ridge Regression baseline on
the supervised feature rows. It uses the same rolling-origin windows and metrics
as the Seasonal Naive baseline, keeps predictions in DataFrames, and remains
local and deterministic.

This Ridge slice does not yet add LightGBM, MLflow tracking, database-backed
training reads, or model artifact persistence. Those pieces build on the Ridge
training and evaluation contract.
```

- [ ] **Step 2: Run the full quality gate**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check . --no-cache
python -m ruff format --check .
python -m pytest
```

Expected: Ruff passes and all tests pass.

- [ ] **Step 3: Commit README documentation**

Run:

```powershell
git add README.md
git commit -m "docs: document ridge baseline modeling"
```

Expected: one documentation commit.

- [ ] **Step 4: Verify implementation branch status before merge**

Run:

```powershell
git status --short --branch
git log --oneline -4
```

Expected: clean working tree on `codex/ridge-baseline-implementation`, with the Ridge feature matrix, Ridge wrapper, Ridge evaluation, and README commits on top of `main`.

- [ ] **Step 5: Merge to main only**

From repository root `D:\Github项目\UrbanFlow-AU`, run:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 fetch origin main
git rev-list --left-right --count main...origin/main
git merge --ff-only codex/ridge-baseline-implementation
```

Expected:

- `git rev-list --left-right --count main...origin/main` prints `0 0` before merge;
- `main` fast-forwards to the local implementation branch.

- [ ] **Step 6: Re-run final checks on main**

Run from `D:\Github项目\UrbanFlow-AU`:

```powershell
$env:PYTHONPATH='src'
python -m ruff check . --no-cache
python -m ruff format --check .
python -m pytest
```

Expected: Ruff passes and all tests pass on `main`.

- [ ] **Step 7: Push main**

Run:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 push origin main
```

Expected: only `main` is pushed to GitHub.

- [ ] **Step 8: Remove the local worktree and local codex branch**

Before removal, verify the resolved path stays under `D:\Github项目\UrbanFlow-AU\.worktrees`:

```powershell
$target = (Resolve-Path '.worktrees\ridge-baseline-implementation').Path
$root = (Resolve-Path '.worktrees').Path
if (-not $target.StartsWith($root)) { throw "Refusing to remove worktree outside .worktrees: $target" }
git worktree remove $target
git worktree prune
git branch -d codex/ridge-baseline-implementation
```

Expected: local implementation worktree and local codex branch are removed after the successful merge and push.

## Self-review checklist

Before executing Task 4 merge/push:

- Spec coverage:
  - scikit-learn dependency: Task 1.
  - feature whitelist and leakage exclusions: Task 1.
  - target filtering and missing-column failures: Task 1.
  - Ridge fit/predict wrapper: Task 2.
  - unknown `location_id` prediction handling: Task 2.
  - windowed training/evaluation filtering: Task 3.
  - rolling-origin Ridge evaluation: Task 3.
  - README documentation: Task 4.
- Placeholder scan:
  - Search this plan for unfinished-marker terms and vague instruction phrases.
  - Any match must be replaced with exact implementation details before execution.
- Type consistency:
  - `ModelTrainingError` is defined once in `feature_matrix.py` and reused by `ridge.py` and `evaluation.py`.
  - `RidgeModelConfig`, `FittedRidgeModel`, `ModelWindowEvaluation`, and `RollingOriginRidgeEvaluation` are exported from `urbanflow.modeling`.
  - The prediction column is consistently `ridge_prediction`.
  - The target column is consistently `target`.
