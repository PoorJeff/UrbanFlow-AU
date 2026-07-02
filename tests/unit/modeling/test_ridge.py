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
