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
