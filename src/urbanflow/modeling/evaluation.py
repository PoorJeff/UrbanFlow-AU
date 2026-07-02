from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from urbanflow.modeling.feature_matrix import ModelTrainingError
from urbanflow.modeling.metrics import (
    RegressionMetrics,
    calculate_regression_metrics,
    summarize_by_group,
)
from urbanflow.modeling.ridge import (
    DEFAULT_RIDGE_MODEL_CONFIG,
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
    model_config: RidgeModelConfig = DEFAULT_RIDGE_MODEL_CONFIG,
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
    model_config: RidgeModelConfig = DEFAULT_RIDGE_MODEL_CONFIG,
) -> RollingOriginRidgeEvaluation:
    validation_windows = tuple(
        evaluate_model_window(supervised_frame, window, model_config=model_config)
        for window in splits.validation_windows
    )
    final_test = evaluate_model_window(
        supervised_frame, splits.final_test, model_config=model_config
    )
    return RollingOriginRidgeEvaluation(
        validation_windows=validation_windows,
        final_test=final_test,
    )
