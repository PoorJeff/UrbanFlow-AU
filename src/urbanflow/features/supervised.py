from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pandas as pd

from urbanflow.features.calendar import add_calendar_features
from urbanflow.features.hourly_panel import WEATHER_COLUMNS, build_hourly_panel
from urbanflow.features.lagged import add_lagged_features

DEFAULT_HORIZONS: tuple[int, ...] = tuple(range(1, 25))


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    parsed_horizons = tuple(int(horizon) for horizon in horizons)
    if not parsed_horizons or any(horizon < 1 or horizon > 24 for horizon in parsed_horizons):
        raise ValueError("horizons must be between 1 and 24")
    return parsed_horizons


def _target_lookup(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[["location_id", "observed_at", "pedestrian_count"]].rename(
        columns={"observed_at": "target_observed_at", "pedestrian_count": "target"}
    )


def build_supervised_frame(
    observations: pd.DataFrame,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    public_holidays: Iterable[date | str] | None = None,
) -> pd.DataFrame:
    parsed_horizons = _validate_horizons(horizons)
    panel = add_lagged_features(build_hourly_panel(observations))
    target_values = _target_lookup(panel)
    horizon_frames: list[pd.DataFrame] = []

    for horizon in parsed_horizons:
        horizon_frame = panel.copy()
        horizon_frame["forecast_origin_at"] = horizon_frame["observed_at"]
        horizon_frame["forecast_horizon"] = horizon
        horizon_frame["target_observed_at"] = horizon_frame["forecast_origin_at"] + pd.Timedelta(
            hours=horizon
        )
        horizon_frame = horizon_frame.merge(
            target_values,
            on=["location_id", "target_observed_at"],
            how="left",
        )
        horizon_frames.append(horizon_frame)

    supervised = pd.concat(horizon_frames, ignore_index=True)
    supervised["target_missing"] = supervised["target"].isna()
    supervised = add_calendar_features(
        supervised,
        timestamp_column="target_observed_at",
        public_holidays=public_holidays,
    )

    for column in WEATHER_COLUMNS:
        if column not in supervised.columns:
            supervised[column] = pd.NA
        marker = f"{column}_missing"
        if marker not in supervised.columns:
            supervised[marker] = supervised[column].isna()

    preferred_columns = [
        "location_id",
        "forecast_origin_at",
        "forecast_horizon",
        "target_observed_at",
        "target",
        "target_missing",
        "pedestrian_count",
        "pedestrian_count_missing",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_24_mean",
        "rolling_24_std",
        "rolling_168_mean",
        "rolling_168_std",
        "hour",
        "weekday",
        "month",
        "is_weekend",
        "is_public_holiday",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "temperature",
        "temperature_missing",
        "rainfall",
        "rainfall_missing",
        "wind_speed",
        "wind_speed_missing",
    ]
    return (
        supervised[preferred_columns]
        .sort_values(["location_id", "forecast_origin_at", "forecast_horizon"])
        .reset_index(drop=True)
    )
