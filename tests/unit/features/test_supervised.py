from __future__ import annotations

from datetime import date

import pandas as pd

from urbanflow.features.supervised import build_supervised_frame


def observations(periods: int = 200) -> pd.DataFrame:
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
            "temperature": [20.0] * periods,
        }
    )


def test_build_supervised_frame_creates_direct_horizon_rows() -> None:
    frame = build_supervised_frame(
        observations(),
        horizons=(1, 2, 24),
        public_holidays={date(2025, 1, 9)},
    )

    origin = pd.Timestamp("2025-01-08 00:00", tz="Australia/Melbourne")
    origin_rows = frame.loc[frame["forecast_origin_at"] == origin].sort_values("forecast_horizon")

    assert origin_rows["forecast_horizon"].tolist() == [1, 2, 24]
    assert origin_rows["target_observed_at"].tolist() == [
        origin + pd.Timedelta(hours=1),
        origin + pd.Timedelta(hours=2),
        origin + pd.Timedelta(hours=24),
    ]
    assert origin_rows["target"].tolist() == [169.0, 170.0, 192.0]
    assert origin_rows["lag_1"].tolist() == [168.0, 168.0, 168.0]
    assert origin_rows["lag_168"].tolist() == [1.0, 1.0, 1.0]
    assert origin_rows["is_public_holiday"].tolist() == [False, False, True]
    assert origin_rows["temperature"].tolist() == [20.0, 20.0, 20.0]
    assert origin_rows["temperature_missing"].tolist() == [False, False, False]
    assert origin_rows["rainfall_missing"].tolist() == [True, True, True]


def test_build_supervised_frame_rejects_empty_or_invalid_horizons() -> None:
    try:
        build_supervised_frame(observations(), horizons=(0, 25))
    except ValueError as exc:
        assert "horizons must be between 1 and 24" in str(exc)
    else:
        raise AssertionError("expected invalid horizon failure")
