from __future__ import annotations

from typing import Final

import pandas as pd

REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"location_id", "observed_at", "pedestrian_count"}
)
WEATHER_COLUMNS: Final[tuple[str, ...]] = ("temperature", "rainfall", "wind_speed")


class FeatureInputError(ValueError):
    """Raised when feature-building input violates the modeling data contract."""


def _validate_required_columns(frame: pd.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise FeatureInputError(f"missing required columns: {', '.join(missing_columns)}")


def _normalise_observed_at(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["observed_at"] = pd.to_datetime(result["observed_at"])
    if result["observed_at"].dt.tz is None:
        raise FeatureInputError("observed_at must be timezone-aware")
    not_hour_boundary = (
        (result["observed_at"].dt.minute != 0)
        | (result["observed_at"].dt.second != 0)
        | (result["observed_at"].dt.microsecond != 0)
        | (result["observed_at"].dt.nanosecond != 0)
    )
    if not_hour_boundary.any():
        raise FeatureInputError("observed_at values must be on an exact hour boundary")
    return result


def validate_hourly_observations(frame: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(frame)
    result = _normalise_observed_at(frame)
    duplicate_mask = result.duplicated(subset=["location_id", "observed_at"], keep=False)
    if duplicate_mask.any():
        raise FeatureInputError("duplicate location_id and observed_at rows are not allowed")
    return result.sort_values(["location_id", "observed_at"]).reset_index(drop=True)


def _complete_sensor_panel(sensor_frame: pd.DataFrame) -> pd.DataFrame:
    sensor_frame = sensor_frame.sort_values("observed_at")
    location_id = int(sensor_frame["location_id"].iloc[0])
    hourly_index = pd.date_range(
        start=sensor_frame["observed_at"].min(),
        end=sensor_frame["observed_at"].max(),
        freq="h",
    )
    completed = (
        sensor_frame.set_index("observed_at")
        .reindex(hourly_index)
        .rename_axis("observed_at")
        .reset_index()
    )
    completed["location_id"] = location_id
    return completed


def build_hourly_panel(frame: pd.DataFrame) -> pd.DataFrame:
    observations = validate_hourly_observations(frame)
    completed_frames = [
        _complete_sensor_panel(sensor_frame)
        for _, sensor_frame in observations.groupby("location_id", sort=True)
    ]
    panel = pd.concat(completed_frames, ignore_index=True)
    panel["pedestrian_count"] = panel["pedestrian_count"].astype("float64")
    panel["pedestrian_count_missing"] = panel["pedestrian_count"].isna()

    for column in WEATHER_COLUMNS:
        if column not in panel.columns:
            panel[column] = pd.NA
        panel[f"{column}_missing"] = panel[column].isna()

    return panel.sort_values(["location_id", "observed_at"]).reset_index(drop=True)
