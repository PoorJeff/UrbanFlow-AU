from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from urbanflow.api.lightgbm_provider import ArtifactBackedLightGBMForecastProvider
from urbanflow.api.postgres import PostgresSensorHistoryRepository
from urbanflow.api.services import (
    DataStoreUnavailableError,
    ForecastBatch,
    ForecastInputUnavailableError,
    ForecastModelOutputError,
)
from urbanflow.database.engine import create_database_engine
from urbanflow.database.models import Base
from urbanflow.database.repositories import upsert_hourly_rows, upsert_sensor_rows
from urbanflow.database.time import MELBOURNE_TZ
from urbanflow.features.supervised import build_supervised_frame
from urbanflow.modeling.feature_matrix import ModelTrainingError
from urbanflow.modeling.lightgbm import LightGBMModelConfig
from urbanflow.modeling.lightgbm_artifact import (
    HolidayCalendar,
    LightGBMArtifactError,
    LightGBMArtifactSerializationError,
    export_lightgbm_artifact,
    load_lightgbm_artifact,
)
from urbanflow.modeling.supervised_csv import read_supervised_csv, sha256_file

SMOKE_DATABASE_URL_ENV_VAR = "URBANFLOW_SMOKE_DATABASE_URL"
SMOKE_LOCATION_ID = 999001

_SAFE_SCHEMA_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_HISTORY_LENGTH = 192
_FORECAST_HORIZON = 24


@dataclass(frozen=True)
class LightGBMForecastSmokeResult:
    schema_name: str
    location_id: int
    data_cutoff_at: str
    forecast_horizons: list[int]
    model_version: str


def validate_smoke_schema_name(schema_name: str) -> str:
    if not _SAFE_SCHEMA_NAME_PATTERN.fullmatch(schema_name):
        raise ValueError(
            "Smoke schema name must be a safe PostgreSQL identifier: "
            "lowercase letters, digits, and underscores only, starting with a letter."
        )
    return schema_name


def run_lightgbm_forecast_smoke(
    database_url: str,
    *,
    schema_name: str | None = None,
) -> LightGBMForecastSmokeResult:
    schema = validate_smoke_schema_name(schema_name or _temporary_schema_name())
    quoted_schema = _quote_identifier(schema)
    engine = create_database_engine(database_url)
    schema_created = False

    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f"CREATE SCHEMA {quoted_schema}")
            schema_created = True
            connection.exec_driver_sql(f"SET search_path TO {quoted_schema}")
            Base.metadata.create_all(connection)
            session_factory = sessionmaker(
                bind=connection,
                autoflush=False,
                expire_on_commit=False,
            )

            with tempfile.TemporaryDirectory(prefix="urbanflow-lightgbm-smoke-") as directory:
                temporary_directory = Path(directory)
                source_timestamps, observations = _smoke_observations()
                calendar = _smoke_holiday_calendar(source_timestamps)
                supervised_csv_path = temporary_directory / "supervised.csv"
                build_supervised_frame(
                    observations,
                    public_holidays=calendar.public_holidays,
                ).to_csv(supervised_csv_path, index=False)
                artifact_path = temporary_directory / "artifact"
                export_lightgbm_artifact(
                    read_supervised_csv(supervised_csv_path),
                    source_csv_sha256=sha256_file(supervised_csv_path),
                    output_directory=artifact_path,
                    holiday_calendar=calendar,
                    model_config=LightGBMModelConfig(
                        n_estimators=5,
                        min_child_samples=1,
                    ),
                )
                artifact = load_lightgbm_artifact(artifact_path)

                with session_factory() as session:
                    upsert_sensor_rows(session, [_sensor_smoke_row()])
                    upsert_hourly_rows(session, _hourly_smoke_rows(source_timestamps))
                    session.commit()

                repository = PostgresSensorHistoryRepository(session_factory)
                provider = ArtifactBackedLightGBMForecastProvider(
                    artifact=artifact,
                    history_repository=repository,
                )
                batch = provider.predict(SMOKE_LOCATION_ID, _FORECAST_HORIZON)
                expected_cutoff = source_timestamps[-1].to_pydatetime()
                _validate_smoke_batch(
                    batch=batch,
                    expected_cutoff=expected_cutoff,
                    expected_model_version=artifact.manifest.model_version,
                )

                return LightGBMForecastSmokeResult(
                    schema_name=schema,
                    location_id=SMOKE_LOCATION_ID,
                    data_cutoff_at=batch.data_cutoff_at.isoformat(),
                    forecast_horizons=[
                        prediction.forecast_horizon for prediction in batch.predictions
                    ],
                    model_version=batch.model_version,
                )
    finally:
        try:
            if schema_created:
                with engine.begin() as connection:
                    connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {quoted_schema} CASCADE")
        finally:
            engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test UrbanFlow AU artifact-backed LightGBM forecasts."
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--schema-name",
        default=None,
        help="Optional temporary schema name for debugging. Defaults to a generated name.",
    )
    return parser


def main(argv: list[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        database_url = _database_url(args.database_url, environ=environ)
        result = run_lightgbm_forecast_smoke(
            database_url,
            schema_name=args.schema_name,
        )
    except (
        LightGBMArtifactError,
        LightGBMArtifactSerializationError,
        ModelTrainingError,
        DataStoreUnavailableError,
        ForecastInputUnavailableError,
        ForecastModelOutputError,
        SQLAlchemyError,
    ) as exc:
        print(f"LightGBM forecast smoke test failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(asdict(result), sort_keys=True))
    return 0


def _temporary_schema_name() -> str:
    return f"urbanflow_lightgbm_smoke_{uuid4().hex[:12]}"


def _quote_identifier(identifier: str) -> str:
    return f'"{validate_smoke_schema_name(identifier)}"'


def _smoke_observations() -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    timestamps = pd.date_range(
        "2026-07-01 00:00",
        periods=_HISTORY_LENGTH,
        freq="h",
        tz=MELBOURNE_TZ,
    )
    counts = [100 + (index % 24) * 3 + index // 24 for index in range(_HISTORY_LENGTH)]
    observations = pd.DataFrame(
        {
            "location_id": [SMOKE_LOCATION_ID] * _HISTORY_LENGTH,
            "observed_at": timestamps,
            "pedestrian_count": counts,
        }
    )
    return timestamps, observations


def _smoke_holiday_calendar(timestamps: pd.DatetimeIndex) -> HolidayCalendar:
    final_target = (
        timestamps[-1].to_pydatetime().astimezone(UTC) + timedelta(hours=_FORECAST_HORIZON)
    ).astimezone(MELBOURNE_TZ)
    return HolidayCalendar(
        coverage_start=timestamps[0].date(),
        coverage_end=final_target.date(),
        public_holidays=(date(2026, 7, 6),),
    )


def _sensor_smoke_row() -> dict[str, object]:
    return {
        "location_id": SMOKE_LOCATION_ID,
        "sensor_name": "LightGBM Forecast Smoke Test Sensor",
        "sensor_description": "Synthetic artifact-backed forecast smoke-test sensor",
        "latitude": -37.8136,
        "longitude": 144.9631,
        "installation_date": date(2026, 7, 1),
        "status": "A",
    }


def _hourly_smoke_rows(timestamps: pd.DatetimeIndex) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, observed_at in enumerate(timestamps):
        pedestrian_count = 100 + (index % 24) * 3 + index // 24
        rows.append(
            {
                "location_id": SMOKE_LOCATION_ID,
                "observed_at": observed_at.to_pydatetime(),
                "source_sensing_date": observed_at.date(),
                "source_hourday": observed_at.hour,
                "pedestrian_count": pedestrian_count,
                "direction_1_count": pedestrian_count // 2,
                "direction_2_count": pedestrian_count - pedestrian_count // 2,
                "source_snapshot_path": "smoke://lightgbm-forecast",
            }
        )
    return rows


def _validate_smoke_batch(
    *,
    batch: ForecastBatch,
    expected_cutoff: datetime,
    expected_model_version: str,
) -> None:
    predictions = batch.predictions
    horizons = [prediction.forecast_horizon for prediction in predictions]
    if horizons != list(range(1, _FORECAST_HORIZON + 1)):
        raise ForecastModelOutputError(f"unexpected forecast horizons: {horizons}")
    if any(
        not math.isfinite(prediction.predicted_count) or prediction.predicted_count < 0
        for prediction in predictions
    ):
        raise ForecastModelOutputError("forecast predictions must be finite and non-negative")
    if batch.data_cutoff_at.astimezone(UTC) != expected_cutoff.astimezone(UTC):
        raise ForecastModelOutputError("forecast cutoff did not match the final source instant")
    if batch.model_version != expected_model_version:
        raise ForecastModelOutputError("forecast model version did not match the artifact")


def _database_url(
    explicit_database_url: str | None,
    *,
    environ: Mapping[str, str] | None,
) -> str:
    values = os.environ if environ is None else environ
    database_url = (
        explicit_database_url
        if explicit_database_url is not None
        else values.get(SMOKE_DATABASE_URL_ENV_VAR)
    )
    if database_url is None or not database_url.strip():
        raise ValueError(
            "LightGBM forecast smoke database URL is required. "
            f"Pass --database-url or set {SMOKE_DATABASE_URL_ENV_VAR}."
        )
    return database_url.strip()
