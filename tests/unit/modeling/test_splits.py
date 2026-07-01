from __future__ import annotations

import pandas as pd
import pytest

from urbanflow.modeling.splits import SplitConfigError, build_rolling_origin_splits


def target_frame(start: str, end: str) -> pd.DataFrame:
    timestamps = pd.date_range(start, end, freq="h", tz="Australia/Melbourne")
    return pd.DataFrame(
        {
            "location_id": [101] * len(timestamps),
            "target_observed_at": timestamps,
            "target": range(len(timestamps)),
        }
    )


def test_build_rolling_origin_splits_uses_final_complete_month_as_test() -> None:
    frame = target_frame("2025-01-01 00:00", "2025-06-30 23:00")

    splits = build_rolling_origin_splits(frame)

    assert splits.final_test.name == "final_test_2025-06"
    assert splits.final_test.start == pd.Timestamp("2025-06-01 00:00", tz="Australia/Melbourne")
    assert splits.final_test.end == pd.Timestamp("2025-07-01 00:00", tz="Australia/Melbourne")
    assert [window.name for window in splits.validation_windows] == [
        "validation_2025-03",
        "validation_2025-04",
        "validation_2025-05",
    ]
    for window in splits.validation_windows:
        assert window.train_end == window.start
        assert window.end <= splits.final_test.start


def test_build_rolling_origin_splits_ignores_incomplete_final_month() -> None:
    frame = target_frame("2025-01-01 00:00", "2025-07-03 12:00")

    splits = build_rolling_origin_splits(frame)

    assert splits.final_test.name == "final_test_2025-06"


def test_build_rolling_origin_splits_requires_two_complete_months() -> None:
    frame = target_frame("2025-01-01 00:00", "2025-01-31 23:00")

    with pytest.raises(SplitConfigError, match="at least two complete months"):
        build_rolling_origin_splits(frame)
