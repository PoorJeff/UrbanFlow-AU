from __future__ import annotations

import pandas as pd

from urbanflow.features.hourly_panel import build_hourly_panel
from urbanflow.features.lagged import add_lagged_features


def hourly_observations(periods: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location_id": [101] * periods,
            "observed_at": pd.date_range(
                "2025-01-01 00:00",
                periods=periods,
                freq="h",
                tz="Australia/Melbourne",
            ),
            "pedestrian_count": list(range(periods)),
        }
    )


def test_add_lagged_features_uses_only_origin_and_prior_counts() -> None:
    panel = build_hourly_panel(hourly_observations(200))

    result = add_lagged_features(panel)
    origin_row = result.loc[result["observed_at"] == panel.loc[167, "observed_at"]].iloc[0]

    assert origin_row["lag_1"] == 167.0
    assert origin_row["lag_24"] == 144.0
    assert origin_row["lag_168"] == 0.0
    assert origin_row["rolling_24_mean"] == sum(range(144, 168)) / 24
    assert round(origin_row["rolling_24_std"], 6) == round(pd.Series(range(144, 168)).std(), 6)
    assert origin_row["rolling_168_mean"] == sum(range(168)) / 168


def test_lagged_features_do_not_change_when_future_counts_change() -> None:
    panel = build_hourly_panel(hourly_observations(220))
    origin = panel.loc[180, "observed_at"]

    baseline = add_lagged_features(panel)
    mutated = panel.copy()
    mutated.loc[mutated["observed_at"] > origin, "pedestrian_count"] = 99999
    after_future_mutation = add_lagged_features(mutated)

    columns = [
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_24_mean",
        "rolling_24_std",
        "rolling_168_mean",
        "rolling_168_std",
    ]
    baseline_row = baseline.loc[baseline["observed_at"] == origin, columns].iloc[0]
    mutated_row = after_future_mutation.loc[
        after_future_mutation["observed_at"] == origin, columns
    ].iloc[0]

    pd.testing.assert_series_equal(baseline_row, mutated_row, check_names=False)
