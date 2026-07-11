from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from urbanflow.modeling.baselines import (
    add_seasonal_naive_predictions,
    derive_seasonal_naive_panel,
)
from urbanflow.modeling.evaluation import (
    _evaluation_rows_for_window,
    _training_rows_for_window,
)
from urbanflow.modeling.lightgbm import (
    DEFAULT_LIGHTGBM_MODEL_CONFIG,
    FittedLightGBMModel,
    LightGBMModelConfig,
    add_lightgbm_predictions,
    fit_lightgbm_model,
)
from urbanflow.modeling.metrics import (
    RegressionMetrics,
    calculate_regression_metrics,
    summarize_by_group,
)
from urbanflow.modeling.splits import EvaluationWindow, RollingOriginSplits


@dataclass(frozen=True)
class LightGBMComparisonMetrics:
    lightgbm_wape: float | None
    seasonal_naive_wape: float | None
    relative_wape_improvement: float | None


@dataclass(frozen=True)
class LightGBMWindowEvaluation:
    window: EvaluationWindow
    predictions: pd.DataFrame
    overall_metrics: RegressionMetrics
    horizon_metrics: pd.DataFrame
    model: FittedLightGBMModel
    seasonal_naive_overall_metrics: RegressionMetrics
    seasonal_naive_horizon_metrics: pd.DataFrame
    model_comparison: LightGBMComparisonMetrics


@dataclass(frozen=True)
class RollingOriginLightGBMEvaluation:
    validation_windows: tuple[LightGBMWindowEvaluation, ...]
    final_test: LightGBMWindowEvaluation


def _lightgbm_comparison_metrics(
    lightgbm_metrics: RegressionMetrics,
    seasonal_naive_metrics: RegressionMetrics,
) -> LightGBMComparisonMetrics:
    lightgbm_wape = lightgbm_metrics.wape
    seasonal_naive_wape = seasonal_naive_metrics.wape
    if lightgbm_wape is None or seasonal_naive_wape in (None, 0):
        improvement = None
    else:
        improvement = (seasonal_naive_wape - lightgbm_wape) / seasonal_naive_wape
    return LightGBMComparisonMetrics(
        lightgbm_wape=lightgbm_wape,
        seasonal_naive_wape=seasonal_naive_wape,
        relative_wape_improvement=improvement,
    )


def evaluate_lightgbm_window(
    supervised_frame: pd.DataFrame,
    window: EvaluationWindow,
    *,
    model_config: LightGBMModelConfig = DEFAULT_LIGHTGBM_MODEL_CONFIG,
    seasonal_naive_panel: pd.DataFrame | None = None,
) -> LightGBMWindowEvaluation:
    training_rows = _training_rows_for_window(supervised_frame, window)
    evaluation_rows = _evaluation_rows_for_window(supervised_frame, window)
    model = fit_lightgbm_model(training_rows, config=model_config)
    predictions = add_lightgbm_predictions(evaluation_rows, model)
    if seasonal_naive_panel is None:
        seasonal_naive_panel = derive_seasonal_naive_panel(supervised_frame)
    predictions = add_seasonal_naive_predictions(predictions, seasonal_naive_panel)
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
    model_comparison = _lightgbm_comparison_metrics(
        overall_metrics,
        seasonal_naive_overall_metrics,
    )
    return LightGBMWindowEvaluation(
        window=window,
        predictions=predictions,
        overall_metrics=overall_metrics,
        horizon_metrics=horizon_metrics,
        model=model,
        seasonal_naive_overall_metrics=seasonal_naive_overall_metrics,
        seasonal_naive_horizon_metrics=seasonal_naive_horizon_metrics,
        model_comparison=model_comparison,
    )


def evaluate_rolling_origin_lightgbm(
    supervised_frame: pd.DataFrame,
    splits: RollingOriginSplits,
    *,
    model_config: LightGBMModelConfig = DEFAULT_LIGHTGBM_MODEL_CONFIG,
) -> RollingOriginLightGBMEvaluation:
    seasonal_naive_panel = derive_seasonal_naive_panel(supervised_frame)
    validation_windows = tuple(
        evaluate_lightgbm_window(
            supervised_frame,
            window,
            model_config=model_config,
            seasonal_naive_panel=seasonal_naive_panel,
        )
        for window in splits.validation_windows
    )
    final_test = evaluate_lightgbm_window(
        supervised_frame,
        splits.final_test,
        model_config=model_config,
        seasonal_naive_panel=seasonal_naive_panel,
    )
    return RollingOriginLightGBMEvaluation(
        validation_windows=validation_windows,
        final_test=final_test,
    )
