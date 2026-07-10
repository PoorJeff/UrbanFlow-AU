from __future__ import annotations

import math

import pandas as pd
import pytest

from urbanflow.modeling.feature_matrix import ModelTrainingError
from urbanflow.modeling.lightgbm import (
    LightGBMModelConfig,
    _clip_nonnegative_predictions,
    add_lightgbm_predictions,
    fit_lightgbm_model,
)


def supervised_rows(periods: int = 48) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01 00:00",
        periods=periods,
        freq="h",
        tz="Australia/Melbourne",
    )
    values = [80.0 + float(index % 24) + float(index // 24) for index in range(periods)]
    return pd.DataFrame(
        {
            "location_id": [101 if index % 2 == 0 else 102 for index in range(periods)],
            "forecast_origin_at": timestamps - pd.Timedelta(hours=1),
            "forecast_horizon": [1 if index % 2 == 0 else 24 for index in range(periods)],
            "target_observed_at": timestamps,
            "target": values,
            "target_missing": [False] * periods,
            "pedestrian_count": [value - 1.0 for value in values],
            "pedestrian_count_missing": [False] * periods,
            "lag_1": [value - 1.0 for value in values],
            "lag_24": [value - 2.0 for value in values],
            "lag_168": [value - 3.0 for value in values],
            "rolling_24_mean": [value - 1.5 for value in values],
            "rolling_24_std": [2.0] * periods,
            "rolling_168_mean": [value - 2.5 for value in values],
            "rolling_168_std": [4.0] * periods,
            "hour": timestamps.hour,
            "weekday": timestamps.weekday,
            "month": timestamps.month,
            "is_weekend": [timestamp.weekday() >= 5 for timestamp in timestamps],
            "is_public_holiday": [False] * periods,
            "hour_sin": [0.1] * periods,
            "hour_cos": [0.9] * periods,
            "weekday_sin": [0.4] * periods,
            "weekday_cos": [0.5] * periods,
            "temperature": [20.0] * periods,
            "temperature_missing": [False] * periods,
            "rainfall": [0.0] * periods,
            "rainfall_missing": [False] * periods,
            "wind_speed": [12.0] * periods,
            "wind_speed_missing": [False] * periods,
        }
    )


def test_fit_lightgbm_model_requires_target() -> None:
    frame = supervised_rows().drop(columns=["target"])

    with pytest.raises(ModelTrainingError, match="missing required columns: target"):
        fit_lightgbm_model(frame)


def test_fit_lightgbm_model_records_training_row_count() -> None:
    model = fit_lightgbm_model(
        supervised_rows(),
        config=LightGBMModelConfig(n_estimators=5, min_child_samples=1),
    )

    assert model.training_row_count == 48


def test_add_lightgbm_predictions_adds_finite_nonnegative_predictions() -> None:
    frame = supervised_rows()
    model = fit_lightgbm_model(
        frame,
        config=LightGBMModelConfig(n_estimators=5, min_child_samples=1),
    )

    result = add_lightgbm_predictions(frame, model)

    assert "lightgbm_prediction" in result.columns
    assert len(result) == len(frame)
    assert result["lightgbm_prediction"].ge(0).all()
    assert all(math.isfinite(value) for value in result["lightgbm_prediction"])


def test_clip_nonnegative_predictions_clips_negative_values() -> None:
    predictions = _clip_nonnegative_predictions([-1.5, 0.0, 2.5])

    assert predictions.tolist() == [0.0, 0.0, 2.5]
