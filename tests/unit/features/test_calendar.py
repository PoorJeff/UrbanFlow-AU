from __future__ import annotations

from datetime import date

import pandas as pd

from urbanflow.features.calendar import add_calendar_features


def test_add_calendar_features_uses_target_timestamp_and_public_holidays() -> None:
    frame = pd.DataFrame(
        {
            "target_observed_at": pd.to_datetime(
                ["2025-01-26 13:00", "2025-01-27 09:00"]
            ).tz_localize("Australia/Melbourne")
        }
    )

    result = add_calendar_features(
        frame,
        timestamp_column="target_observed_at",
        public_holidays={date(2025, 1, 27)},
    )

    assert result["hour"].tolist() == [13, 9]
    assert result["weekday"].tolist() == [6, 0]
    assert result["month"].tolist() == [1, 1]
    assert result["is_weekend"].tolist() == [True, False]
    assert result["is_public_holiday"].tolist() == [False, True]
    assert result["hour_sin"].round(6).tolist() == [-0.258819, 0.707107]
    assert result["hour_cos"].round(6).tolist() == [-0.965926, -0.707107]
    assert result["weekday_sin"].round(6).tolist() == [-0.781831, 0.0]
    assert result["weekday_cos"].round(6).tolist() == [0.62349, 1.0]


def test_add_calendar_features_rejects_timezone_naive_timestamp() -> None:
    frame = pd.DataFrame({"target_observed_at": pd.to_datetime(["2025-01-01 00:00"])})

    try:
        add_calendar_features(frame, timestamp_column="target_observed_at")
    except ValueError as exc:
        assert "timezone-aware" in str(exc)
    else:
        raise AssertionError("expected timezone-aware validation failure")
