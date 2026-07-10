from __future__ import annotations

import pandas as pd


class SeasonalNaiveBaselineError(ValueError):
    """Raised when Seasonal Naive baseline inputs cannot be derived."""


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

    return (
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
