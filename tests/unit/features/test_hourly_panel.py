from __future__ import annotations

import pandas as pd
import pytest

from urbanflow.features.hourly_panel import FeatureInputError, build_hourly_panel


def melbourne_range(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="h", tz="Australia/Melbourne")


def test_build_hourly_panel_fills_missing_sensor_hours_without_imputation() -> None:
    timestamps = melbourne_range("2025-01-01 00:00", 4)
    frame = pd.DataFrame(
        {
            "location_id": [101, 101, 101],
            "observed_at": [timestamps[0], timestamps[2], timestamps[3]],
            "pedestrian_count": [10, 30, 40],
        }
    )

    panel = build_hourly_panel(frame)

    assert panel["location_id"].tolist() == [101, 101, 101, 101]
    assert panel["observed_at"].tolist() == list(timestamps)
    assert panel["pedestrian_count"].tolist()[:1] == [10.0]
    assert pd.isna(panel.loc[1, "pedestrian_count"])
    assert panel["pedestrian_count"].tolist()[2:] == [30.0, 40.0]
    assert panel["pedestrian_count_missing"].tolist() == [False, True, False, False]


def test_build_hourly_panel_preserves_optional_weather_and_missing_markers() -> None:
    timestamps = melbourne_range("2025-01-01 00:00", 2)
    frame = pd.DataFrame(
        {
            "location_id": [101, 101],
            "observed_at": list(timestamps),
            "pedestrian_count": [10, 20],
            "temperature": [21.5, None],
            "rainfall": [0.0, 1.2],
            "wind_speed": [None, 8.0],
        }
    )

    panel = build_hourly_panel(frame)

    assert panel["temperature"].tolist()[0] == 21.5
    assert pd.isna(panel.loc[1, "temperature"])
    assert panel["temperature_missing"].tolist() == [False, True]
    assert panel["rainfall_missing"].tolist() == [False, False]
    assert panel["wind_speed_missing"].tolist() == [True, False]


def test_build_hourly_panel_rejects_duplicate_sensor_timestamp() -> None:
    timestamp = melbourne_range("2025-01-01 00:00", 1)[0]
    frame = pd.DataFrame(
        {
            "location_id": [101, 101],
            "observed_at": [timestamp, timestamp],
            "pedestrian_count": [10, 11],
        }
    )

    with pytest.raises(FeatureInputError, match="duplicate"):
        build_hourly_panel(frame)


def test_build_hourly_panel_rejects_non_hour_boundary() -> None:
    frame = pd.DataFrame(
        {
            "location_id": [101],
            "observed_at": [pd.Timestamp("2025-01-01 00:30", tz="Australia/Melbourne")],
            "pedestrian_count": [10],
        }
    )

    with pytest.raises(FeatureInputError, match="hour boundary"):
        build_hourly_panel(frame)
