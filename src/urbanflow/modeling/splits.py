from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SplitConfigError(ValueError):
    """Raised when chronological split windows cannot be derived."""


@dataclass(frozen=True)
class EvaluationWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    train_end: pd.Timestamp


@dataclass(frozen=True)
class RollingOriginSplits:
    validation_windows: tuple[EvaluationWindow, ...]
    final_test: EvaluationWindow


def _ensure_timestamp_series(frame: pd.DataFrame, *, timestamp_column: str) -> pd.Series:
    if timestamp_column not in frame.columns:
        raise SplitConfigError(f"missing timestamp column: {timestamp_column}")
    timestamps = pd.to_datetime(frame[timestamp_column])
    if timestamps.dt.tz is None:
        raise SplitConfigError(f"{timestamp_column} must be timezone-aware")
    return timestamps


def _month_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(
        year=timestamp.year,
        month=timestamp.month,
        day=1,
        tz=timestamp.tz,
    )


def _next_month_start(month_start: pd.Timestamp) -> pd.Timestamp:
    if month_start.month == 12:
        return pd.Timestamp(year=month_start.year + 1, month=1, day=1, tz=month_start.tz)
    return pd.Timestamp(
        year=month_start.year,
        month=month_start.month + 1,
        day=1,
        tz=month_start.tz,
    )


def _month_label(month_start: pd.Timestamp) -> str:
    return f"{month_start.year:04d}-{month_start.month:02d}"


def complete_months(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "target_observed_at",
) -> tuple[pd.Timestamp, ...]:
    timestamps = _ensure_timestamp_series(frame, timestamp_column=timestamp_column)
    unique_timestamps = set(timestamps.dropna().tolist())
    first_month = _month_start(timestamps.min())
    last_month = _month_start(timestamps.max())

    months: list[pd.Timestamp] = []
    current_month = first_month
    while current_month <= last_month:
        next_month = _next_month_start(current_month)
        expected = set(
            pd.date_range(
                current_month,
                next_month - pd.Timedelta(hours=1),
                freq="h",
            )
        )
        if expected.issubset(unique_timestamps):
            months.append(current_month)
        current_month = next_month
    return tuple(months)


def build_rolling_origin_splits(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "target_observed_at",
    validation_months: int = 3,
) -> RollingOriginSplits:
    if validation_months < 1:
        raise SplitConfigError("validation_months must be at least 1")

    months = complete_months(frame, timestamp_column=timestamp_column)
    if len(months) < 2:
        raise SplitConfigError("at least two complete months are required")

    final_test_start = months[-1]
    final_test_end = _next_month_start(final_test_start)
    final_test = EvaluationWindow(
        name=f"final_test_{_month_label(final_test_start)}",
        start=final_test_start,
        end=final_test_end,
        train_end=final_test_start,
    )

    validation_starts = months[max(0, len(months) - 1 - validation_months) : -1]
    validation_windows = tuple(
        EvaluationWindow(
            name=f"validation_{_month_label(month_start)}",
            start=month_start,
            end=_next_month_start(month_start),
            train_end=month_start,
        )
        for month_start in validation_starts
    )
    return RollingOriginSplits(validation_windows=validation_windows, final_test=final_test)
