# Leakage-Safe Features and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build leakage-safe feature engineering, direct multi-horizon supervised rows, rolling-origin split utilities, metrics, and a Seasonal Naive baseline.

**Architecture:** Add pure pandas-based `urbanflow.features` and `urbanflow.modeling` packages. Keep the first modeling foundation independent of PostgreSQL, live network calls, MLflow, Ridge, and LightGBM so the data contract and leakage tests are proven before model training begins.

**Tech Stack:** Python 3.11+, pandas, pytest, Ruff, standard-library dataclasses/math/zoneinfo.

---

## Source spec

Implement the approved design in:

`docs/superpowers/specs/2026-06-30-leakage-safe-features-evaluation-design.md`

This plan intentionally does not add Ridge Regression, LightGBM, MLflow,
weather ingestion, public-holiday downloads, API routes, dashboards, or SQL
feature marts.

## File structure

- Create `src/urbanflow/features/__init__.py`
  - Export feature errors and public feature-building helpers.
- Create `src/urbanflow/features/calendar.py`
  - Add target-time calendar, weekend, public-holiday, and cyclic encodings.
- Create `src/urbanflow/features/hourly_panel.py`
  - Validate input hourly observations and build complete per-sensor panels.
- Create `src/urbanflow/features/lagged.py`
  - Add origin-anchored lag and rolling features.
- Create `src/urbanflow/features/supervised.py`
  - Build direct `forecast_horizon=1..24` supervised rows.
- Create `src/urbanflow/modeling/__init__.py`
  - Export split, metric, and baseline helpers.
- Create `src/urbanflow/modeling/splits.py`
  - Define chronological rolling-origin validation and final-test windows.
- Create `src/urbanflow/modeling/metrics.py`
  - Compute aggregate and grouped regression metrics.
- Create `src/urbanflow/modeling/baselines.py`
  - Add Seasonal Naive predictions and coverage-aware evaluation.
- Create `tests/unit/features/test_calendar.py`
- Create `tests/unit/features/test_hourly_panel.py`
- Create `tests/unit/features/test_lagged.py`
- Create `tests/unit/features/test_supervised.py`
- Create `tests/unit/modeling/test_splits.py`
- Create `tests/unit/modeling/test_metrics.py`
- Create `tests/unit/modeling/test_baselines.py`

## Shared implementation conventions

Use these names consistently across all tasks:

- input count column: `pedestrian_count`
- sensor key: `location_id`
- raw observation timestamp: `observed_at`
- forecast origin timestamp: `forecast_origin_at`
- target timestamp: `target_observed_at`
- horizon column: `forecast_horizon`
- target value: `target`
- target missing marker: `target_missing`
- observation missing marker: `pedestrian_count_missing`
- weather columns: `temperature`, `rainfall`, `wind_speed`
- weather missing markers: `temperature_missing`, `rainfall_missing`, `wind_speed_missing`
- Seasonal Naive prediction column: `seasonal_naive_prediction`

Use `ValueError` subclasses instead of bare `ValueError` for domain failures:

```python
class FeatureInputError(ValueError):
    """Raised when feature-building input violates the modeling data contract."""
```

```python
class SplitConfigError(ValueError):
    """Raised when chronological split windows cannot be derived."""
```

## Task 1: Calendar features and complete hourly panels

**Files:**
- Create: `src/urbanflow/features/__init__.py`
- Create: `src/urbanflow/features/calendar.py`
- Create: `src/urbanflow/features/hourly_panel.py`
- Create: `tests/unit/features/test_calendar.py`
- Create: `tests/unit/features/test_hourly_panel.py`

- [ ] **Step 1: Write failing calendar tests**

Create `tests/unit/features/test_calendar.py`:

```python
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
    frame = pd.DataFrame(
        {"target_observed_at": pd.to_datetime(["2025-01-01 00:00"])}
    )

    try:
        add_calendar_features(frame, timestamp_column="target_observed_at")
    except ValueError as exc:
        assert "timezone-aware" in str(exc)
    else:
        raise AssertionError("expected timezone-aware validation failure")
```

- [ ] **Step 2: Write failing hourly panel tests**

Create `tests/unit/features/test_hourly_panel.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/features/test_calendar.py tests/unit/features/test_hourly_panel.py -v
```

Expected: FAIL during collection because `urbanflow.features` does not exist.

- [ ] **Step 4: Implement calendar helpers**

Create `src/urbanflow/features/calendar.py`:

```python
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
    result["weekday_sin"] = result["weekday"].map(
        lambda value: math.sin(2 * math.pi * value / 7)
    )
    result["weekday_cos"] = result["weekday"].map(
        lambda value: math.cos(2 * math.pi * value / 7)
    )
    return result
```

- [ ] **Step 5: Implement hourly panel builder**

Create `src/urbanflow/features/hourly_panel.py`:

```python
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
```

- [ ] **Step 6: Export feature helpers**

Create `src/urbanflow/features/__init__.py`:

```python
from urbanflow.features.calendar import add_calendar_features
from urbanflow.features.hourly_panel import FeatureInputError, build_hourly_panel

__all__ = [
    "FeatureInputError",
    "add_calendar_features",
    "build_hourly_panel",
]
```

- [ ] **Step 7: Run focused tests to verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/features/test_calendar.py tests/unit/features/test_hourly_panel.py -v
```

Expected: all tests in these two files pass.

- [ ] **Step 8: Run focused Ruff checks**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check src/urbanflow/features tests/unit/features --no-cache
python -m ruff format --check src/urbanflow/features tests/unit/features
```

Expected: no Ruff diagnostics and no formatting changes required.

- [ ] **Step 9: Commit Task 1**

Run:

```powershell
git add src/urbanflow/features tests/unit/features/test_calendar.py tests/unit/features/test_hourly_panel.py
git commit -m "feat: add hourly feature panel foundation"
```

Expected: one commit containing the new `urbanflow.features` package foundation and its tests.

## Task 2: Origin-anchored lag/rolling features and supervised horizon rows

**Files:**
- Create: `src/urbanflow/features/lagged.py`
- Create: `src/urbanflow/features/supervised.py`
- Modify: `src/urbanflow/features/__init__.py`
- Create: `tests/unit/features/test_lagged.py`
- Create: `tests/unit/features/test_supervised.py`

- [ ] **Step 1: Write failing lag/rolling tests**

Create `tests/unit/features/test_lagged.py`:

```python
from __future__ import annotations

import pandas as pd

from urbanflow.features.hourly_panel import build_hourly_panel
from urbanflow.features.lagged import add_lagged_features


def hourly_observations(periods: int) -> pd.DataFrame:
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


def test_add_lagged_features_uses_only_origin_and_prior_counts() -> None:
    panel = build_hourly_panel(hourly_observations(200))

    result = add_lagged_features(panel)
    origin_row = result.loc[result["observed_at"] == panel.loc[167, "observed_at"]].iloc[0]

    assert origin_row["lag_1"] == 167.0
    assert origin_row["lag_24"] == 144.0
    assert origin_row["lag_168"] == 0.0
    assert origin_row["rolling_24_mean"] == sum(range(144, 168)) / 24
    assert round(origin_row["rolling_24_std"], 6) == round(pd.Series(range(144, 168)).std(), 6)
    assert origin_row["rolling_168_mean"] == sum(range(168)) / 168


def test_lagged_features_do_not_change_when_future_counts_change() -> None:
    panel = build_hourly_panel(hourly_observations(220))
    origin = panel.loc[180, "observed_at"]

    baseline = add_lagged_features(panel)
    mutated = panel.copy()
    mutated.loc[mutated["observed_at"] > origin, "pedestrian_count"] = 99999
    after_future_mutation = add_lagged_features(mutated)

    columns = [
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_24_mean",
        "rolling_24_std",
        "rolling_168_mean",
        "rolling_168_std",
    ]
    baseline_row = baseline.loc[baseline["observed_at"] == origin, columns].iloc[0]
    mutated_row = after_future_mutation.loc[
        after_future_mutation["observed_at"] == origin, columns
    ].iloc[0]

    pd.testing.assert_series_equal(baseline_row, mutated_row, check_names=False)
```

- [ ] **Step 2: Write failing supervised frame tests**

Create `tests/unit/features/test_supervised.py`:

```python
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
    origin_rows = frame.loc[frame["forecast_origin_at"] == origin].sort_values(
        "forecast_horizon"
    )

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
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/features/test_lagged.py tests/unit/features/test_supervised.py -v
```

Expected: FAIL during collection because `urbanflow.features.lagged` and
`urbanflow.features.supervised` do not exist.

- [ ] **Step 4: Implement lagged features**

Create `src/urbanflow/features/lagged.py`:

```python
from __future__ import annotations

import pandas as pd

from urbanflow.features.hourly_panel import FeatureInputError


def add_lagged_features(panel: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"location_id", "observed_at", "pedestrian_count"}
    missing_columns = sorted(required_columns.difference(panel.columns))
    if missing_columns:
        raise FeatureInputError(f"missing required columns: {', '.join(missing_columns)}")

    result = panel.sort_values(["location_id", "observed_at"]).reset_index(drop=True).copy()
    grouped_counts = result.groupby("location_id", sort=False)["pedestrian_count"]

    result["lag_1"] = grouped_counts.shift(0)
    result["lag_24"] = grouped_counts.shift(23)
    result["lag_168"] = grouped_counts.shift(167)
    result["rolling_24_mean"] = grouped_counts.transform(
        lambda series: series.rolling(window=24, min_periods=24).mean()
    )
    result["rolling_24_std"] = grouped_counts.transform(
        lambda series: series.rolling(window=24, min_periods=24).std()
    )
    result["rolling_168_mean"] = grouped_counts.transform(
        lambda series: series.rolling(window=168, min_periods=168).mean()
    )
    result["rolling_168_std"] = grouped_counts.transform(
        lambda series: series.rolling(window=168, min_periods=168).std()
    )
    return result
```

- [ ] **Step 5: Implement supervised frame builder**

Create `src/urbanflow/features/supervised.py`:

```python
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
    return supervised[preferred_columns].sort_values(
        ["location_id", "forecast_origin_at", "forecast_horizon"]
    ).reset_index(drop=True)
```

- [ ] **Step 6: Export lagged and supervised helpers**

Modify `src/urbanflow/features/__init__.py`:

```python
from urbanflow.features.calendar import add_calendar_features
from urbanflow.features.hourly_panel import FeatureInputError, build_hourly_panel
from urbanflow.features.lagged import add_lagged_features
from urbanflow.features.supervised import build_supervised_frame

__all__ = [
    "FeatureInputError",
    "add_calendar_features",
    "add_lagged_features",
    "build_hourly_panel",
    "build_supervised_frame",
]
```

- [ ] **Step 7: Run focused tests to verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/features -v
```

Expected: all feature tests pass.

- [ ] **Step 8: Run focused Ruff checks**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check src/urbanflow/features tests/unit/features --no-cache
python -m ruff format --check src/urbanflow/features tests/unit/features
```

Expected: no Ruff diagnostics and no formatting changes required.

- [ ] **Step 9: Commit Task 2**

Run:

```powershell
git add src/urbanflow/features tests/unit/features
git commit -m "feat: add leakage-safe supervised feature rows"
```

Expected: one commit containing lag/rolling features, supervised rows, exports, and tests.

## Task 3: Chronological rolling-origin split utilities

**Files:**
- Create: `src/urbanflow/modeling/__init__.py`
- Create: `src/urbanflow/modeling/splits.py`
- Create: `tests/unit/modeling/test_splits.py`

- [ ] **Step 1: Write failing split tests**

Create `tests/unit/modeling/test_splits.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/modeling/test_splits.py -v
```

Expected: FAIL during collection because `urbanflow.modeling` does not exist.

- [ ] **Step 3: Implement split utilities**

Create `src/urbanflow/modeling/splits.py`:

```python
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
    return pd.Timestamp(year=month_start.year, month=month_start.month + 1, day=1, tz=month_start.tz)


def _month_label(month_start: pd.Timestamp) -> str:
    return f"{month_start.year:04d}-{month_start.month:02d}"


def complete_months(frame: pd.DataFrame, *, timestamp_column: str = "target_observed_at") -> tuple[pd.Timestamp, ...]:
    timestamps = _ensure_timestamp_series(frame, timestamp_column=timestamp_column)
    unique_timestamps = set(timestamps.dropna().tolist())
    first_month = _month_start(timestamps.min())
    last_month = _month_start(timestamps.max())

    months: list[pd.Timestamp] = []
    current_month = first_month
    while current_month <= last_month:
        next_month = _next_month_start(current_month)
        expected = set(pd.date_range(current_month, next_month - pd.Timedelta(hours=1), freq="h"))
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
```

- [ ] **Step 4: Export modeling split helpers**

Create `src/urbanflow/modeling/__init__.py`:

```python
from urbanflow.modeling.splits import (
    EvaluationWindow,
    RollingOriginSplits,
    SplitConfigError,
    build_rolling_origin_splits,
    complete_months,
)

__all__ = [
    "EvaluationWindow",
    "RollingOriginSplits",
    "SplitConfigError",
    "build_rolling_origin_splits",
    "complete_months",
]
```

- [ ] **Step 5: Run split tests to verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/modeling/test_splits.py -v
```

Expected: all split tests pass.

- [ ] **Step 6: Run focused Ruff checks**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check src/urbanflow/modeling tests/unit/modeling/test_splits.py --no-cache
python -m ruff format --check src/urbanflow/modeling tests/unit/modeling/test_splits.py
```

Expected: no Ruff diagnostics and no formatting changes required.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add src/urbanflow/modeling tests/unit/modeling/test_splits.py
git commit -m "feat: add rolling-origin split utilities"
```

Expected: one commit containing modeling package exports, split utilities, and split tests.

## Task 4: Metrics and Seasonal Naive baseline

**Files:**
- Create: `src/urbanflow/modeling/metrics.py`
- Create: `src/urbanflow/modeling/baselines.py`
- Modify: `src/urbanflow/modeling/__init__.py`
- Create: `tests/unit/modeling/test_metrics.py`
- Create: `tests/unit/modeling/test_baselines.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/unit/modeling/test_metrics.py`:

```python
from __future__ import annotations

import pandas as pd

from urbanflow.modeling.metrics import (
    calculate_regression_metrics,
    peak_top_decile_mae,
    summarize_by_group,
)


def test_calculate_regression_metrics_returns_mae_rmse_wape_and_row_count() -> None:
    metrics = calculate_regression_metrics(
        actual=pd.Series([10.0, 20.0, 30.0]),
        predicted=pd.Series([12.0, 18.0, 33.0]),
    )

    assert metrics.row_count == 3
    assert metrics.mae == 7 / 3
    assert round(metrics.rmse, 6) == 2.380476
    assert metrics.wape == 7 / 60


def test_calculate_regression_metrics_returns_none_wape_for_zero_denominator() -> None:
    metrics = calculate_regression_metrics(
        actual=pd.Series([0.0, 0.0]),
        predicted=pd.Series([1.0, 2.0]),
    )

    assert metrics.wape is None


def test_summarize_by_group_computes_per_sensor_metrics() -> None:
    frame = pd.DataFrame(
        {
            "location_id": [101, 101, 202],
            "target": [10.0, 20.0, 30.0],
            "prediction": [12.0, 18.0, 33.0],
        }
    )

    summary = summarize_by_group(
        frame,
        group_columns=("location_id",),
        actual_column="target",
        prediction_column="prediction",
    )

    assert summary.loc[summary["location_id"] == 101, "wape"].iloc[0] == 4 / 30
    assert summary.loc[summary["location_id"] == 202, "mae"].iloc[0] == 3


def test_peak_top_decile_mae_uses_actual_values_to_select_peaks() -> None:
    frame = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 300.0],
            "prediction": [1.0, 2.0, 3.0, 90.0, 110.0, 130.0, 150.0, 170.0, 190.0, 270.0],
        }
    )

    assert peak_top_decile_mae(frame, actual_column="target", prediction_column="prediction") == 30.0
```

- [ ] **Step 2: Write failing Seasonal Naive tests**

Create `tests/unit/modeling/test_baselines.py`:

```python
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
        result["target_observed_at"]
        == pd.Timestamp("2025-01-08 01:00", tz="Australia/Melbourne")
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
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/modeling/test_metrics.py tests/unit/modeling/test_baselines.py -v
```

Expected: FAIL during collection because `urbanflow.modeling.metrics` and
`urbanflow.modeling.baselines` do not exist.

- [ ] **Step 4: Implement metric helpers**

Create `src/urbanflow/modeling/metrics.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    row_count: int
    mae: float | None
    rmse: float | None
    wape: float | None


def _valid_metric_rows(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": actual, "predicted": predicted})
    return frame.dropna(subset=["actual", "predicted"])


def calculate_regression_metrics(actual: pd.Series, predicted: pd.Series) -> RegressionMetrics:
    rows = _valid_metric_rows(actual, predicted)
    if rows.empty:
        return RegressionMetrics(row_count=0, mae=None, rmse=None, wape=None)

    absolute_errors = (rows["actual"] - rows["predicted"]).abs()
    squared_errors = (rows["actual"] - rows["predicted"]) ** 2
    denominator = rows["actual"].abs().sum()
    return RegressionMetrics(
        row_count=len(rows),
        mae=float(absolute_errors.mean()),
        rmse=float(math.sqrt(squared_errors.mean())),
        wape=None if denominator == 0 else float(absolute_errors.sum() / denominator),
    )


def summarize_by_group(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    actual_column: str,
    prediction_column: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        metrics = calculate_regression_metrics(group[actual_column], group[prediction_column])
        record = dict(zip(group_columns, group_key, strict=True))
        record.update(
            {
                "row_count": metrics.row_count,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "wape": metrics.wape,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def peak_top_decile_mae(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    prediction_column: str,
) -> float | None:
    rows = frame.dropna(subset=[actual_column, prediction_column])
    if rows.empty:
        return None
    peak_count = max(1, math.ceil(len(rows) * 0.10))
    peak_rows = rows.nlargest(peak_count, actual_column)
    metrics = calculate_regression_metrics(peak_rows[actual_column], peak_rows[prediction_column])
    return metrics.mae
```

- [ ] **Step 5: Implement Seasonal Naive helper**

Create `src/urbanflow/modeling/baselines.py`:

```python
from __future__ import annotations

import pandas as pd


def add_seasonal_naive_predictions(
    supervised_frame: pd.DataFrame,
    panel_frame: pd.DataFrame,
    *,
    prediction_column: str = "seasonal_naive_prediction",
) -> pd.DataFrame:
    result = supervised_frame.copy()
    history = panel_frame[["location_id", "observed_at", "pedestrian_count"]].rename(
        columns={
            "observed_at": "seasonal_naive_observed_at",
            "pedestrian_count": prediction_column,
        }
    )
    result["seasonal_naive_observed_at"] = result["target_observed_at"] - pd.Timedelta(hours=168)
    result = result.merge(
        history,
        on=["location_id", "seasonal_naive_observed_at"],
        how="left",
    )
    result["seasonal_naive_missing"] = result[prediction_column].isna()
    return result
```

- [ ] **Step 6: Export metric and baseline helpers**

Modify `src/urbanflow/modeling/__init__.py`:

```python
from urbanflow.modeling.baselines import add_seasonal_naive_predictions
from urbanflow.modeling.metrics import (
    RegressionMetrics,
    calculate_regression_metrics,
    peak_top_decile_mae,
    summarize_by_group,
)
from urbanflow.modeling.splits import (
    EvaluationWindow,
    RollingOriginSplits,
    SplitConfigError,
    build_rolling_origin_splits,
    complete_months,
)

__all__ = [
    "EvaluationWindow",
    "RegressionMetrics",
    "RollingOriginSplits",
    "SplitConfigError",
    "add_seasonal_naive_predictions",
    "build_rolling_origin_splits",
    "calculate_regression_metrics",
    "complete_months",
    "peak_top_decile_mae",
    "summarize_by_group",
]
```

- [ ] **Step 7: Run modeling tests to verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/unit/modeling -v
```

Expected: all modeling tests pass.

- [ ] **Step 8: Run focused Ruff checks**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check src/urbanflow/modeling tests/unit/modeling --no-cache
python -m ruff format --check src/urbanflow/modeling tests/unit/modeling
```

Expected: no Ruff diagnostics and no formatting changes required.

- [ ] **Step 9: Commit Task 4**

Run:

```powershell
git add src/urbanflow/modeling tests/unit/modeling
git commit -m "feat: add seasonal naive evaluation foundation"
```

Expected: one commit containing metrics, baseline helper, exports, and tests.

## Task 5: Full verification, README note, merge, push, and cleanup

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a short README section for the modeling foundation**

Add this section after the existing local Prefect ingestion flow documentation:

```markdown
## Build leakage-safe modeling features

The first modeling foundation is intentionally local and deterministic. It
builds supervised `forecast_horizon=1..24` rows from hourly pedestrian
observations, adds calendar, lag, rolling, missing-marker, and optional weather
columns, and evaluates a Seasonal Naive baseline through chronological split
utilities.

The implementation is DataFrame-first so it can be tested without PostgreSQL,
network access, MLflow, Ridge, or LightGBM. Subsequent modeling slices will add
database readers, Ridge, LightGBM, and MLflow tracking on top of the same
feature and split contracts.
```

- [ ] **Step 2: Run the full quality gate**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check . --no-cache
python -m ruff format --check .
python -m pytest
```

Expected: Ruff passes and all tests pass.

- [ ] **Step 3: Commit README documentation**

Run:

```powershell
git add README.md
git commit -m "docs: document leakage-safe modeling foundation"
```

Expected: one documentation commit.

- [ ] **Step 4: Verify branch status before merge**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree on `codex/leakage-safe-features-evaluation`.

- [ ] **Step 5: Merge to main only**

From repository root `D:\Github项目\UrbanFlow-AU`, run:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 fetch origin main
git merge --ff-only codex/leakage-safe-features-evaluation
```

Expected: `main` fast-forwards to the local implementation branch.

- [ ] **Step 6: Re-run final checks on main**

Run:

```powershell
$env:PYTHONPATH='src'
python -m ruff check . --no-cache
python -m ruff format --check .
python -m pytest
```

Expected: Ruff passes and all tests pass on `main`.

- [ ] **Step 7: Push main**

Run:

```powershell
git -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 push origin main
```

Expected: only `main` is pushed to GitHub.

- [ ] **Step 8: Remove the local worktree and local codex branch**

Before removal, verify the resolved path stays under
`D:\Github项目\UrbanFlow-AU\.worktrees`:

```powershell
$target = (Resolve-Path '.worktrees\leakage-safe-features-evaluation').Path
$root = (Resolve-Path '.worktrees').Path
if (-not $target.StartsWith($root)) { throw "Refusing to remove worktree outside .worktrees: $target" }
git worktree remove $target
git worktree prune
git branch -d codex/leakage-safe-features-evaluation
```

Expected: local feature worktree and local codex branch are removed after the successful merge and push.

## Self-review checklist

- Spec coverage:
  - Time/calendar/cyclic/public-holiday features: Task 1.
  - Complete hourly panel and missing markers: Task 1.
  - `lag_1`, `lag_24`, `lag_168`, rolling 24/168 mean/std: Task 2.
  - Direct `forecast_horizon=1..24` supervised rows: Task 2.
  - Weather columns and missing markers without weather ingestion: Tasks 1 and 2.
  - Rolling-origin validation and final-test windows: Task 3.
  - MAE, RMSE, WAPE, per-sensor/grouped metrics, peak Top 10% MAE: Task 4.
  - Seasonal Naive baseline: Task 4.
  - No Ridge, LightGBM, MLflow, API, dashboard, public-holiday download, or weather ingestion: all tasks.
- Unresolved-marker scan:
  - The plan contains no unresolved markers.
  - Each code-changing task names exact files, exact commands, and expected outcomes.
- Type consistency:
  - `FeatureInputError`, `SplitConfigError`, `RegressionMetrics`,
    `EvaluationWindow`, and `RollingOriginSplits` are defined before export.
  - Test imports match the modules created by each task.
  - Column names match the source spec and shared implementation conventions.
