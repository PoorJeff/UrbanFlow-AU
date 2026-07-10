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
    assert result.model.training_row_count == 172
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
    assert result.model_comparison.seasonal_naive_wape == result.seasonal_naive_overall_metrics.wape


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
    assert result.validation_windows[0].seasonal_naive_overall_metrics.row_count > 0
    assert result.final_test.seasonal_naive_overall_metrics.row_count > 0
