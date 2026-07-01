from __future__ import annotations

import pandas as pd

from urbanflow.modeling.metrics import (
    calculate_regression_metrics,
    peak_top_decile_mae,
    summarize_by_group,
)


def test_calculate_regression_metrics_returns_mae_rmse_wape_and_row_count() -> None:
    metrics = calculate_regression_metrics(
        actual=pd.Series([10.0, 20.0, 30.0]),
        predicted=pd.Series([12.0, 18.0, 33.0]),
    )

    assert metrics.row_count == 3
    assert metrics.mae == 7 / 3
    assert round(metrics.rmse, 6) == 2.380476
    assert metrics.wape == 7 / 60


def test_calculate_regression_metrics_returns_none_wape_for_zero_denominator() -> None:
    metrics = calculate_regression_metrics(
        actual=pd.Series([0.0, 0.0]),
        predicted=pd.Series([1.0, 2.0]),
    )

    assert metrics.wape is None


def test_summarize_by_group_computes_per_sensor_metrics() -> None:
    frame = pd.DataFrame(
        {
            "location_id": [101, 101, 202],
            "target": [10.0, 20.0, 30.0],
            "prediction": [12.0, 18.0, 33.0],
        }
    )

    summary = summarize_by_group(
        frame,
        group_columns=("location_id",),
        actual_column="target",
        prediction_column="prediction",
    )

    assert summary.loc[summary["location_id"] == 101, "wape"].iloc[0] == 4 / 30
    assert summary.loc[summary["location_id"] == 202, "mae"].iloc[0] == 3


def test_peak_top_decile_mae_uses_actual_values_to_select_peaks() -> None:
    frame = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 300.0],
            "prediction": [1.0, 2.0, 3.0, 90.0, 110.0, 130.0, 150.0, 170.0, 190.0, 270.0],
        }
    )

    assert (
        peak_top_decile_mae(
            frame,
            actual_column="target",
            prediction_column="prediction",
        )
        == 30.0
    )
