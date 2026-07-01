from __future__ import annotations

import pandas as pd

from urbanflow.features.hourly_panel import FeatureInputError


def add_lagged_features(panel: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"location_id", "observed_at", "pedestrian_count"}
    missing_columns = sorted(required_columns.difference(panel.columns))
    if missing_columns:
        raise FeatureInputError(f"missing required columns: {', '.join(missing_columns)}")

    result = panel.sort_values(["location_id", "observed_at"]).reset_index(drop=True).copy()
    grouped_counts = result.groupby("location_id", sort=False)["pedestrian_count"]

    result["lag_1"] = grouped_counts.shift(0)
    result["lag_24"] = grouped_counts.shift(23)
    result["lag_168"] = grouped_counts.shift(167)
    result["rolling_24_mean"] = grouped_counts.transform(
        lambda series: series.rolling(window=24, min_periods=24).mean()
    )
    result["rolling_24_std"] = grouped_counts.transform(
        lambda series: series.rolling(window=24, min_periods=24).std()
    )
    result["rolling_168_mean"] = grouped_counts.transform(
        lambda series: series.rolling(window=168, min_periods=168).mean()
    )
    result["rolling_168_std"] = grouped_counts.transform(
        lambda series: series.rolling(window=168, min_periods=168).std()
    )
    return result
