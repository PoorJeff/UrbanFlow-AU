from __future__ import annotations

import pandas as pd


def add_seasonal_naive_predictions(
    supervised_frame: pd.DataFrame,
    panel_frame: pd.DataFrame,
    *,
    prediction_column: str = "seasonal_naive_prediction",
) -> pd.DataFrame:
    result = supervised_frame.copy()
    history = panel_frame[["location_id", "observed_at", "pedestrian_count"]].rename(
        columns={
            "observed_at": "seasonal_naive_observed_at",
            "pedestrian_count": prediction_column,
        }
    )
    result["seasonal_naive_observed_at"] = result["target_observed_at"] - pd.Timedelta(hours=168)
    result = result.merge(
        history,
        on=["location_id", "seasonal_naive_observed_at"],
        how="left",
    )
    result["seasonal_naive_missing"] = result[prediction_column].isna()
    return result
