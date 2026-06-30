from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import date

import pandas as pd


def _normalise_holidays(public_holidays: Iterable[date | str] | None) -> set[date]:
    if public_holidays is None:
        return set()
    return {
        holiday if isinstance(holiday, date) else date.fromisoformat(str(holiday))
        for holiday in public_holidays
    }


def _ensure_timezone_aware(series: pd.Series, *, column: str) -> pd.Series:
    timestamps = pd.to_datetime(series)
    if timestamps.dt.tz is None:
        raise ValueError(f"{column} must be timezone-aware")
    return timestamps


def add_calendar_features(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    public_holidays: Iterable[date | str] | None = None,
) -> pd.DataFrame:
    if timestamp_column not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_column}")

    result = frame.copy()
    timestamps = _ensure_timezone_aware(result[timestamp_column], column=timestamp_column)
    holidays = _normalise_holidays(public_holidays)

    result["hour"] = timestamps.dt.hour
    result["weekday"] = timestamps.dt.weekday
    result["month"] = timestamps.dt.month
    result["is_weekend"] = result["weekday"].isin([5, 6])
    result["is_public_holiday"] = timestamps.dt.date.isin(holidays)
    result["hour_sin"] = result["hour"].map(lambda value: math.sin(2 * math.pi * value / 24))
    result["hour_cos"] = result["hour"].map(lambda value: math.cos(2 * math.pi * value / 24))
    result["weekday_sin"] = result["weekday"].map(lambda value: math.sin(2 * math.pi * value / 7))
    result["weekday_cos"] = result["weekday"].map(lambda value: math.cos(2 * math.pi * value / 7))
    return result
