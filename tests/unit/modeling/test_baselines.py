from __future__ import annotations

import pandas as pd

from urbanflow.features.hourly_panel import build_hourly_panel
from urbanflow.features.supervised import build_supervised_frame
from urbanflow.modeling.baselines import add_seasonal_naive_predictions


def observations(periods: int) -> pd.DataFrame:
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


def test_add_seasonal_naive_predictions_uses_same_hour_one_week_prior() -> None:
    source = observations(220)
    supervised = build_supervised_frame(source, horizons=(1,))
    panel = build_hourly_panel(source)

    result = add_seasonal_naive_predictions(supervised, panel)
    row = result.loc[
        result["target_observed_at"] == pd.Timestamp("2025-01-08 01:00", tz="Australia/Melbourne")
    ].iloc[0]

    assert row["target"] == 169.0
    assert row["seasonal_naive_prediction"] == 1.0
    assert not bool(row["seasonal_naive_missing"])


def test_add_seasonal_naive_predictions_marks_missing_history() -> None:
    source = observations(10)
    supervised = build_supervised_frame(source, horizons=(1,))
    panel = build_hourly_panel(source)

    result = add_seasonal_naive_predictions(supervised, panel)

    assert result["seasonal_naive_prediction"].isna().all()
    assert result["seasonal_naive_missing"].all()
