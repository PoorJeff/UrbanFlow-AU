from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ModelTrainingError(ValueError):
    """Raised when model training or prediction inputs are invalid."""


@dataclass(frozen=True)
class ModelFeatureSpec:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...] = ("location_id",)
    target_column: str = "target"
    target_missing_column: str = "target_missing"

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.categorical_columns

    @property
    def required_training_columns(self) -> tuple[str, ...]:
        return self.feature_columns + (self.target_column, self.target_missing_column)


DEFAULT_RIDGE_FEATURE_SPEC = ModelFeatureSpec(
    numeric_columns=(
        "forecast_horizon",
        "pedestrian_count",
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
        "pedestrian_count_missing",
    ),
    categorical_columns=("location_id",),
)


def _missing_columns(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(column for column in required_columns if column not in frame.columns)


def validate_model_feature_columns(
    frame: pd.DataFrame,
    *,
    feature_spec: ModelFeatureSpec = DEFAULT_RIDGE_FEATURE_SPEC,
    require_target: bool = True,
) -> None:
    required_columns = (
        feature_spec.required_training_columns if require_target else feature_spec.feature_columns
    )
    missing = _missing_columns(frame, required_columns)
    if missing:
        raise ModelTrainingError(f"missing required columns: {', '.join(missing)}")


def select_training_rows(
    frame: pd.DataFrame,
    *,
    feature_spec: ModelFeatureSpec = DEFAULT_RIDGE_FEATURE_SPEC,
) -> pd.DataFrame:
    validate_model_feature_columns(frame, feature_spec=feature_spec, require_target=True)
    target_missing = frame[feature_spec.target_column].isna() | frame[
        feature_spec.target_missing_column
    ].fillna(True).astype(bool)
    rows = frame.loc[~target_missing].copy()
    if rows.empty:
        raise ModelTrainingError("no training rows remain after filtering missing targets")
    return rows


def select_model_features(
    frame: pd.DataFrame,
    *,
    feature_spec: ModelFeatureSpec = DEFAULT_RIDGE_FEATURE_SPEC,
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    if require_target:
        rows = select_training_rows(frame, feature_spec=feature_spec)
        target = rows[feature_spec.target_column].astype(float)
    else:
        validate_model_feature_columns(frame, feature_spec=feature_spec, require_target=False)
        rows = frame.copy()
        target = None
    return rows[list(feature_spec.feature_columns)].copy(), target
