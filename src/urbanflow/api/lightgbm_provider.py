from __future__ import annotations

from datetime import UTC, datetime, timedelta
from numbers import Integral

import pandas as pd

from urbanflow.api.services import (
    ForecastBatch,
    ForecastInputUnavailableError,
    ForecastModelOutputError,
    ForecastPrediction,
    HistoryRecord,
    RecentHistoryRepository,
)
from urbanflow.database.time import MELBOURNE_TZ
from urbanflow.features.supervised import build_supervised_frame
from urbanflow.modeling.lightgbm_artifact import HolidayCalendar, LoadedLightGBMArtifact

RECENT_HISTORY_LIMIT = 168
MAX_FORECAST_HORIZON = 24


def _validate_horizon(horizon: int) -> None:
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, Integral)
        or horizon < 1
        or horizon > MAX_FORECAST_HORIZON
    ):
        raise ForecastInputUnavailableError("forecast horizon is invalid")


def _history_to_observations(
    location_id: int,
    records: list[HistoryRecord],
) -> pd.DataFrame:
    if len(records) != RECENT_HISTORY_LIMIT:
        raise ForecastInputUnavailableError("history must contain exactly 168 records")

    observations: list[dict[str, object]] = []
    previous_instant: datetime | None = None
    for record in records:
        observed_at = record.observed_at
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ForecastInputUnavailableError("history timestamp is timezone-naive")
        instant = observed_at.astimezone(UTC)
        if any(
            (
                instant.minute,
                instant.second,
                instant.microsecond,
                getattr(instant, "nanosecond", 0),
            )
        ):
            raise ForecastInputUnavailableError("history timestamp is not an exact hour")
        if previous_instant is not None and instant - previous_instant != timedelta(hours=1):
            raise ForecastInputUnavailableError("history is not contiguous")

        count = record.pedestrian_count
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ForecastInputUnavailableError("history count is invalid")
        observations.append(
            {
                "location_id": location_id,
                "observed_at": instant.astimezone(MELBOURNE_TZ),
                "pedestrian_count": int(count),
            }
        )
        previous_instant = instant

    return pd.DataFrame(observations)


def _target_at(*, cutoff: datetime, step: int) -> datetime:
    return (cutoff.astimezone(UTC) + timedelta(hours=step)).astimezone(MELBOURNE_TZ)


def _validate_target_calendar(
    calendar: HolidayCalendar,
    *,
    cutoff: datetime,
    horizon: int,
) -> None:
    for step in range(1, horizon + 1):
        if not calendar.contains(_target_at(cutoff=cutoff, step=step).date()):
            raise ForecastInputUnavailableError("holiday calendar does not cover target date")


def _forecast_rows(
    observations: pd.DataFrame,
    *,
    cutoff: datetime,
    horizon: int,
    calendar: HolidayCalendar,
) -> pd.DataFrame:
    supervised = build_supervised_frame(
        observations,
        horizons=range(1, horizon + 1),
        public_holidays=calendar.public_holidays,
    )
    rows = supervised.loc[supervised["forecast_origin_at"] == pd.Timestamp(cutoff)].sort_values(
        "forecast_horizon"
    )
    if rows["forecast_horizon"].tolist() != list(range(1, horizon + 1)):
        raise ForecastInputUnavailableError("could not construct direct forecast rows")

    expected_targets = [
        pd.Timestamp(_target_at(cutoff=cutoff, step=step)) for step in range(1, horizon + 1)
    ]
    if rows["target_observed_at"].tolist() != expected_targets:
        raise ForecastInputUnavailableError("direct forecast targets are not instant-contiguous")
    return rows


class ArtifactBackedLightGBMForecastProvider:
    def __init__(
        self,
        *,
        artifact: LoadedLightGBMArtifact,
        history_repository: RecentHistoryRepository,
    ) -> None:
        self._artifact = artifact
        self._history_repository = history_repository

    def predict(self, location_id: int, horizon: int) -> ForecastBatch:
        _validate_horizon(horizon)
        records = self._history_repository.get_recent_history(
            location_id,
            limit=RECENT_HISTORY_LIMIT,
        )
        observations = _history_to_observations(location_id, records)
        cutoff = records[-1].observed_at.astimezone(MELBOURNE_TZ)
        calendar = self._artifact.manifest.holiday_calendar
        _validate_target_calendar(calendar, cutoff=cutoff, horizon=horizon)
        rows = _forecast_rows(
            observations,
            cutoff=cutoff,
            horizon=horizon,
            calendar=calendar,
        )

        raw_predictions = self._artifact.model.predict(rows)
        try:
            predicted_values = tuple(float(value) for value in raw_predictions)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ForecastModelOutputError("model predictions are not numeric") from exc
        if len(predicted_values) != len(rows):
            raise ForecastModelOutputError("model returned an invalid prediction count")
        predictions = tuple(
            ForecastPrediction(
                forecast_horizon=int(row.forecast_horizon),
                target_at=pd.Timestamp(row.target_observed_at).to_pydatetime(),
                predicted_count=predicted_count,
            )
            for row, predicted_count in zip(
                rows.itertuples(index=False),
                predicted_values,
                strict=True,
            )
        )
        return ForecastBatch(
            model_name="lightgbm",
            model_version=self._artifact.manifest.model_version,
            generated_at=datetime.now(UTC),
            forecast_origin_at=cutoff,
            data_cutoff_at=cutoff,
            predictions=predictions,
        )
