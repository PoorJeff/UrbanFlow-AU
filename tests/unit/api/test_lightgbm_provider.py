from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import urbanflow.api.lightgbm_provider as lightgbm_provider_module
from tests.unit.api.helpers import InMemorySensorRepository, make_sensor
from urbanflow.api.errors import UrbanFlowApiError
from urbanflow.api.lightgbm_provider import ArtifactBackedLightGBMForecastProvider
from urbanflow.api.services import (
    DataStoreUnavailableError,
    ForecastInputUnavailableError,
    ForecastService,
    HistoryRecord,
)
from urbanflow.database.time import MELBOURNE_TZ
from urbanflow.modeling.feature_matrix import DEFAULT_RIDGE_FEATURE_SPEC
from urbanflow.modeling.lightgbm import FittedLightGBMModel, LightGBMModelConfig
from urbanflow.modeling.lightgbm_artifact import (
    HolidayCalendar,
    LightGBMArtifactManifest,
    LoadedLightGBMArtifact,
    export_lightgbm_artifact,
    load_lightgbm_artifact,
)
from urbanflow.modeling.supervised_csv import read_supervised_csv, sha256_file


@dataclass
class RecordingRecentHistoryRepository:
    records: list[HistoryRecord]
    error: DataStoreUnavailableError | None = None
    calls: list[tuple[int, int]] = field(default_factory=list)

    def get_recent_history(
        self,
        location_id: int,
        *,
        limit: int,
    ) -> list[HistoryRecord]:
        self.calls.append((location_id, limit))
        if self.error is not None:
            raise self.error
        return self.records


@dataclass
class RecordingModel:
    predictions: tuple[float, ...]
    calls: list[pd.DataFrame] = field(default_factory=list)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.calls.append(frame.copy())
        return np.asarray(self.predictions, dtype=float)


def _history(
    *,
    end: datetime = datetime(2026, 7, 12, 10, tzinfo=UTC),
    length: int = 168,
) -> list[HistoryRecord]:
    start = end - timedelta(hours=length - 1)
    return [
        HistoryRecord(
            observed_at=start + timedelta(hours=index),
            pedestrian_count=100 + index,
        )
        for index in range(length)
    ]


def _manifest(calendar: HolidayCalendar) -> LightGBMArtifactManifest:
    return LightGBMArtifactManifest(
        schema_version=1,
        model_name="lightgbm",
        model_version="lightgbm-test-v1",
        model_sha256="0" * 64,
        training_data_sha256="1" * 64,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        trained_through_at=datetime(2025, 12, 31, tzinfo=UTC),
        training_row_count=192,
        feature_timezone="Australia/Melbourne",
        feature_columns=DEFAULT_RIDGE_FEATURE_SPEC.feature_columns,
        model_config={
            "n_estimators": 5,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 1,
            "random_state": 42,
        },
        holiday_calendar=calendar,
        evaluation_summary_path=None,
    )


def _recording_artifact(
    model: RecordingModel,
    *,
    calendar: HolidayCalendar | None = None,
) -> LoadedLightGBMArtifact:
    selected_calendar = calendar or HolidayCalendar(
        coverage_start=date(2026, 1, 1),
        coverage_end=date(2026, 12, 31),
        public_holidays=(),
    )
    return LoadedLightGBMArtifact(
        manifest=_manifest(selected_calendar),
        model=cast(FittedLightGBMModel, model),
    )


def _training_rows() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-04-05 00:00",
        periods=192,
        freq="h",
        tz="Australia/Melbourne",
    )
    index = pd.Series(range(len(timestamps)), dtype="float64")
    return pd.DataFrame(
        {
            "location_id": [101] * len(timestamps),
            "forecast_origin_at": timestamps,
            "forecast_horizon": [1] * len(timestamps),
            "target_observed_at": timestamps + pd.Timedelta(1, unit="h"),
            "target": 100.0 + index,
            "target_missing": [False] * len(timestamps),
            "pedestrian_count": 99.0 + index,
            "pedestrian_count_missing": [False] * len(timestamps),
            "lag_1": 98.0 + index,
            "lag_24": 97.0 + index,
            "lag_168": 96.0 + index,
            "rolling_24_mean": 95.0 + index,
            "rolling_24_std": [2.0] * len(timestamps),
            "rolling_168_mean": 94.0 + index,
            "rolling_168_std": [4.0] * len(timestamps),
            "hour": timestamps.hour,
            "weekday": timestamps.weekday,
            "month": timestamps.month,
            "is_weekend": [value.weekday() >= 5 for value in timestamps],
            "is_public_holiday": [False] * len(timestamps),
            "hour_sin": [math.sin((value.hour / 24.0) * math.tau) for value in timestamps],
            "hour_cos": [math.cos((value.hour / 24.0) * math.tau) for value in timestamps],
            "weekday_sin": [math.sin((value.weekday() / 7.0) * math.tau) for value in timestamps],
            "weekday_cos": [math.cos((value.weekday() / 7.0) * math.tau) for value in timestamps],
            "temperature": [pd.NA] * len(timestamps),
            "temperature_missing": [True] * len(timestamps),
            "rainfall": [pd.NA] * len(timestamps),
            "rainfall_missing": [True] * len(timestamps),
            "wind_speed": [pd.NA] * len(timestamps),
            "wind_speed_missing": [True] * len(timestamps),
        }
    )


def _real_artifact(tmp_path: Path) -> LoadedLightGBMArtifact:
    csv_path = tmp_path / "supervised.csv"
    _training_rows().to_csv(csv_path, index=False)
    calendar = HolidayCalendar(
        coverage_start=date(2026, 1, 1),
        coverage_end=date(2026, 12, 31),
        public_holidays=(date(2026, 7, 13),),
    )
    artifact_path = tmp_path / "artifact"
    export_lightgbm_artifact(
        read_supervised_csv(csv_path),
        source_csv_sha256=sha256_file(csv_path),
        output_directory=artifact_path,
        holiday_calendar=calendar,
        model_config=LightGBMModelConfig(n_estimators=5, min_child_samples=1),
    )
    return load_lightgbm_artifact(artifact_path)


def test_predict_returns_a_direct_batch_from_a_real_artifact(tmp_path: Path) -> None:
    history = _history()
    repository = RecordingRecentHistoryRepository(records=history)
    artifact = _real_artifact(tmp_path)
    provider = ArtifactBackedLightGBMForecastProvider(
        artifact=artifact,
        history_repository=repository,
    )

    batch = provider.predict(location_id=101, horizon=24)

    cutoff = history[-1].observed_at.astimezone(MELBOURNE_TZ)
    assert repository.calls == [(101, 168)]
    assert batch.model_name == "lightgbm"
    assert batch.model_version == artifact.manifest.model_version
    assert batch.data_cutoff_at == cutoff
    assert batch.forecast_origin_at == cutoff
    assert [item.forecast_horizon for item in batch.predictions] == list(range(1, 25))
    assert [item.target_at for item in batch.predictions] == [
        (cutoff.astimezone(UTC) + timedelta(hours=step)).astimezone(MELBOURNE_TZ)
        for step in range(1, 25)
    ]
    assert batch.generated_at.tzinfo is UTC


def test_predict_builds_direct_features_across_melbourne_dst_fallback() -> None:
    history = _history(end=datetime(2026, 4, 4, 14, tzinfo=UTC))
    original_history = list(history)
    repository = RecordingRecentHistoryRepository(records=history)
    model = RecordingModel(predictions=(-1.5, 2.5, 3.5))
    calendar = HolidayCalendar(
        coverage_start=date(2026, 4, 5),
        coverage_end=date(2026, 4, 5),
        public_holidays=(date(2026, 4, 5),),
    )
    provider = ArtifactBackedLightGBMForecastProvider(
        artifact=_recording_artifact(model, calendar=calendar),
        history_repository=repository,
    )

    batch = provider.predict(location_id=101, horizon=3)

    assert history == original_history
    assert len(history) == 168
    assert len(model.calls) == 1
    rows = model.calls[0]
    cutoff = history[-1].observed_at.astimezone(MELBOURNE_TZ)
    expected_targets = [
        (cutoff.astimezone(UTC) + timedelta(hours=step)).astimezone(MELBOURNE_TZ)
        for step in range(1, 4)
    ]
    assert rows["forecast_horizon"].tolist() == [1, 2, 3]
    assert rows["forecast_origin_at"].tolist() == [pd.Timestamp(cutoff)] * 3
    assert rows["target_observed_at"].tolist() == [
        pd.Timestamp(value) for value in expected_targets
    ]
    assert rows["is_public_holiday"].tolist() == [True, True, True]
    for value_column in ("temperature", "rainfall", "wind_speed"):
        assert rows[value_column].isna().all()
        assert rows[f"{value_column}_missing"].tolist() == [True, True, True]
    assert [item.target_at for item in batch.predictions] == expected_targets
    assert [item.predicted_count for item in batch.predictions] == [-1.5, 2.5, 3.5]
    assert [
        later.astimezone(UTC) - earlier.astimezone(UTC)
        for earlier, later in zip(expected_targets, expected_targets[1:], strict=False)
    ] == [timedelta(hours=1), timedelta(hours=1)]
    assert [value.hour for value in expected_targets] == [2, 2, 3]
    assert [value.utcoffset() for value in expected_targets] == [
        timedelta(hours=11),
        timedelta(hours=10),
        timedelta(hours=10),
    ]


def test_history_normalization_preserves_both_fallback_two_oclock_instants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = _history(end=datetime(2026, 4, 4, 16, tzinfo=UTC))
    repository = RecordingRecentHistoryRepository(records=history)
    model = RecordingModel(predictions=(1.0,))
    captured_observations: list[pd.DataFrame] = []
    real_builder = lightgbm_provider_module.build_supervised_frame

    def recording_builder(
        observations: pd.DataFrame,
        **kwargs: object,
    ) -> pd.DataFrame:
        captured_observations.append(observations.copy())
        return real_builder(observations, **kwargs)

    monkeypatch.setattr(
        lightgbm_provider_module,
        "build_supervised_frame",
        recording_builder,
    )
    provider = ArtifactBackedLightGBMForecastProvider(
        artifact=_recording_artifact(
            model,
            calendar=HolidayCalendar(
                coverage_start=date(2026, 4, 5),
                coverage_end=date(2026, 4, 5),
                public_holidays=(),
            ),
        ),
        history_repository=repository,
    )

    provider.predict(location_id=101, horizon=1)

    assert len(captured_observations) == 1
    normalized = captured_observations[0]["observed_at"].tolist()
    fallback_hours = [
        timestamp
        for timestamp in normalized
        if timestamp.date() == date(2026, 4, 5) and timestamp.hour == 2
    ]
    assert len(fallback_hours) == 2
    assert [timestamp.utcoffset() for timestamp in fallback_hours] == [
        timedelta(hours=11),
        timedelta(hours=10),
    ]
    assert [timestamp.astimezone(UTC) for timestamp in fallback_hours] == [
        pd.Timestamp("2026-04-04T15:00:00Z"),
        pd.Timestamp("2026-04-04T16:00:00Z"),
    ]
    assert fallback_hours[1].astimezone(UTC) - fallback_hours[0].astimezone(UTC) == timedelta(
        hours=1
    )
    assert model.calls[0]["forecast_origin_at"].iloc[0] == pd.Timestamp(fallback_hours[-1])


def _invalid_history(case: str) -> list[HistoryRecord]:
    records = _history()
    if case == "167-records":
        return records[1:]
    if case == "169-records":
        return [
            HistoryRecord(
                observed_at=records[0].observed_at - timedelta(hours=1),
                pedestrian_count=99,
            ),
            *records,
        ]
    if case == "reverse-order":
        return list(reversed(records))
    if case == "gap":
        records[80] = HistoryRecord(
            observed_at=records[80].observed_at + timedelta(hours=1),
            pedestrian_count=records[80].pedestrian_count,
        )
    elif case == "naive-timestamp":
        records[80] = HistoryRecord(
            observed_at=records[80].observed_at.replace(tzinfo=None),
            pedestrian_count=records[80].pedestrian_count,
        )
    elif case == "minute-30":
        records[80] = HistoryRecord(
            observed_at=records[80].observed_at + timedelta(minutes=30),
            pedestrian_count=records[80].pedestrian_count,
        )
    elif case == "nanosecond":
        return [
            HistoryRecord(
                observed_at=pd.Timestamp(record.observed_at) + pd.Timedelta(1, unit="ns"),
                pedestrian_count=record.pedestrian_count,
            )
            for record in records
        ]
    elif case == "bool-count":
        records[80] = HistoryRecord(
            observed_at=records[80].observed_at,
            pedestrian_count=cast(int, True),
        )
    elif case == "float-count":
        records[80] = HistoryRecord(
            observed_at=records[80].observed_at,
            pedestrian_count=cast(int, 1.5),
        )
    elif case == "negative-count":
        records[80] = HistoryRecord(
            observed_at=records[80].observed_at,
            pedestrian_count=-1,
        )
    else:
        raise AssertionError(f"unknown case: {case}")
    return records


@pytest.mark.parametrize(
    "case",
    [
        "167-records",
        "169-records",
        "reverse-order",
        "gap",
        "naive-timestamp",
        "minute-30",
        "nanosecond",
        "bool-count",
        "float-count",
        "negative-count",
    ],
)
def test_invalid_history_fails_before_model_prediction(case: str) -> None:
    model = RecordingModel(predictions=(1.0,))
    repository = RecordingRecentHistoryRepository(records=_invalid_history(case))
    provider = ArtifactBackedLightGBMForecastProvider(
        artifact=_recording_artifact(model),
        history_repository=repository,
    )

    with pytest.raises(ForecastInputUnavailableError):
        provider.predict(location_id=101, horizon=1)

    assert repository.calls == [(101, 168)]
    assert model.calls == []


def test_uncovered_target_date_fails_before_model_prediction() -> None:
    model = RecordingModel(predictions=(1.0,))
    calendar = HolidayCalendar(
        coverage_start=date(2026, 7, 12),
        coverage_end=date(2026, 7, 12),
        public_holidays=(),
    )
    provider = ArtifactBackedLightGBMForecastProvider(
        artifact=_recording_artifact(model, calendar=calendar),
        history_repository=RecordingRecentHistoryRepository(
            records=_history(end=datetime(2026, 7, 12, 13, tzinfo=UTC))
        ),
    )

    with pytest.raises(ForecastInputUnavailableError, match="holiday calendar"):
        provider.predict(location_id=101, horizon=1)

    assert model.calls == []


@pytest.mark.parametrize("horizon", [True, 0, 25, 1.5, "1"])
def test_invalid_horizon_fails_before_repository_or_model(horizon: object) -> None:
    model = RecordingModel(predictions=(1.0,))
    repository = RecordingRecentHistoryRepository(records=_history())
    provider = ArtifactBackedLightGBMForecastProvider(
        artifact=_recording_artifact(model),
        history_repository=repository,
    )

    with pytest.raises(ForecastInputUnavailableError, match="horizon"):
        provider.predict(location_id=101, horizon=cast(int, horizon))

    assert repository.calls == []
    assert model.calls == []


def test_data_store_failure_propagates_unchanged() -> None:
    error = DataStoreUnavailableError("database unavailable")
    repository = RecordingRecentHistoryRepository(records=[], error=error)
    provider = ArtifactBackedLightGBMForecastProvider(
        artifact=_recording_artifact(RecordingModel(predictions=(1.0,))),
        history_repository=repository,
    )

    with pytest.raises(DataStoreUnavailableError) as caught:
        provider.predict(location_id=101, horizon=1)

    assert caught.value is error
    assert repository.calls == [(101, 168)]


def test_short_model_output_remains_a_model_unavailable_error() -> None:
    model = RecordingModel(predictions=(1.0,))
    provider = ArtifactBackedLightGBMForecastProvider(
        artifact=_recording_artifact(model),
        history_repository=RecordingRecentHistoryRepository(records=_history()),
    )
    service = ForecastService(
        sensor_repository=InMemorySensorRepository(records=[make_sensor()]),
        model_provider=provider,
    )

    with pytest.raises(UrbanFlowApiError) as caught:
        service.forecast(location_id=101, horizon=2)

    assert caught.value.status_code == 503
    assert caught.value.code == "model_unavailable"
    assert len(model.calls) == 1
