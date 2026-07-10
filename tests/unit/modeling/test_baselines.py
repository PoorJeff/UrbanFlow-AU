from __future__ import annotations

import pandas as pd
import pytest

from urbanflow.features.hourly_panel import build_hourly_panel
from urbanflow.features.supervised import build_supervised_frame
from urbanflow.modeling.baselines import (
    SeasonalNaiveBaselineError,
    add_seasonal_naive_predictions,
    derive_seasonal_naive_panel,
)


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


def test_derive_seasonal_naive_panel_deduplicates_matching_targets() -> None:
    timestamps = pd.to_datetime(
        [
            "2025-01-01 00:00",
            "2025-01-01 00:00",
            "2025-01-01 01:00",
        ],
        utc=True,
    )
    supervised = pd.DataFrame(
        {
            "location_id": [101, 101, 101],
            "target_observed_at": timestamps,
            "target": [10.0, 10.0, 12.0],
        }
    )

    panel = derive_seasonal_naive_panel(supervised)

    assert list(panel.columns) == ["location_id", "observed_at", "pedestrian_count"]
    assert len(panel) == 2
    assert panel.to_dict(orient="records") == [
        {
            "location_id": 101,
            "observed_at": timestamps[0],
            "pedestrian_count": 10.0,
        },
        {
            "location_id": 101,
            "observed_at": timestamps[2],
            "pedestrian_count": 12.0,
        },
    ]


def test_derive_seasonal_naive_panel_rejects_conflicting_duplicate_targets() -> None:
    timestamp = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    supervised = pd.DataFrame(
        {
            "location_id": [101, 101],
            "target_observed_at": [timestamp, timestamp],
            "target": [10.0, 11.0],
        }
    )

    with pytest.raises(
        SeasonalNaiveBaselineError,
        match="conflicting target values for duplicate location_id and target_observed_at",
    ):
        derive_seasonal_naive_panel(supervised)


def test_derive_seasonal_naive_panel_requires_input_columns() -> None:
    supervised = pd.DataFrame({"location_id": [101], "target": [10.0]})

    with pytest.raises(
        SeasonalNaiveBaselineError,
        match="missing required Seasonal Naive input columns: target_observed_at",
    ):
        derive_seasonal_naive_panel(supervised)
