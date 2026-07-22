from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from urbanflow.modeling.feature_matrix import (
    DEFAULT_RIDGE_FEATURE_SPEC,
    ModelTrainingError,
    select_training_rows,
)
from urbanflow.modeling.lightgbm import (
    FittedLightGBMModel,
    LightGBMModelConfig,
    fit_lightgbm_model,
)
from urbanflow.modeling.supervised_csv import sha256_file

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_MODEL_NAME = "lightgbm"
ARTIFACT_MODEL_FILE_NAME = "model.joblib"
ARTIFACT_MANIFEST_FILE_NAME = "manifest.json"
FEATURE_TIMEZONE = "Australia/Melbourne"
_EXPECTED_BUNDLE_FILES = {
    ARTIFACT_MANIFEST_FILE_NAME,
    ARTIFACT_MODEL_FILE_NAME,
}
_MANIFEST_KEYS = {
    "schema_version",
    "model_name",
    "model_version",
    "model_sha256",
    "training_data_sha256",
    "created_at",
    "trained_through_at",
    "training_row_count",
    "feature_timezone",
    "feature_columns",
    "model_config",
    "holiday_calendar_start",
    "holiday_calendar_end",
    "public_holidays",
    "evaluation_summary_path",
}
_MODEL_CONFIG_KEYS = {
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "random_state",
}
_WEATHER_CONTRACT = (
    ("temperature", "temperature_missing"),
    ("rainfall", "rainfall_missing"),
    ("wind_speed", "wind_speed_missing"),
)
_LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_URI_SCHEME_PREFIX = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")


class LightGBMArtifactError(ValueError):
    """Raised when a LightGBM artifact input or bundle is invalid."""


class LightGBMArtifactSerializationError(RuntimeError):
    """Raised when a validated artifact cannot be serialized atomically."""


@dataclass(frozen=True, slots=True)
class HolidayCalendar:
    coverage_start: date
    coverage_end: date
    public_holidays: tuple[date, ...]

    @classmethod
    def from_json_file(cls, path: Path) -> HolidayCalendar:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LightGBMArtifactError(f"invalid holiday calendar: {path}") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "coverage_start",
            "coverage_end",
            "public_holidays",
        }:
            raise LightGBMArtifactError(f"invalid holiday calendar: {path}")
        try:
            calendar = cls(
                coverage_start=_parse_iso_date(payload["coverage_start"]),
                coverage_end=_parse_iso_date(payload["coverage_end"]),
                public_holidays=tuple(
                    _parse_iso_date(value) for value in _require_list(payload["public_holidays"])
                ),
            )
            _validate_holiday_calendar(calendar)
        except (TypeError, ValueError) as exc:
            raise LightGBMArtifactError(f"invalid holiday calendar: {path}") from exc
        return calendar

    def contains(self, value: date) -> bool:
        return self.coverage_start <= value <= self.coverage_end

    def to_manifest_fields(self) -> dict[str, object]:
        return {
            "holiday_calendar_start": self.coverage_start.isoformat(),
            "holiday_calendar_end": self.coverage_end.isoformat(),
            "public_holidays": [value.isoformat() for value in self.public_holidays],
        }


@dataclass(frozen=True, slots=True)
class LightGBMArtifactManifest:
    schema_version: int
    model_name: str
    model_version: str
    model_sha256: str
    training_data_sha256: str
    created_at: datetime
    trained_through_at: datetime
    training_row_count: int
    feature_timezone: str
    feature_columns: tuple[str, ...]
    model_config: dict[str, int | float]
    holiday_calendar: HolidayCalendar
    evaluation_summary_path: str | None


@dataclass(frozen=True, slots=True)
class LoadedLightGBMArtifact:
    manifest: LightGBMArtifactManifest
    model: FittedLightGBMModel


def _validated_local_path(raw_path: str | Path) -> Path:
    try:
        raw_text = os.fspath(raw_path)
    except TypeError as exc:
        raise LightGBMArtifactError("artifact location must be a local path") from exc
    if not isinstance(raw_text, str) or not raw_text.strip() or "://" in raw_text:
        raise LightGBMArtifactError("artifact location must be a local path")
    path = Path(raw_path)
    scheme_match = _URI_SCHEME_PREFIX.match(str(path))
    if scheme_match and len(scheme_match.group(1)) != 1:
        raise LightGBMArtifactError("artifact location must be a local path")
    return path


def _parse_iso_date(value: object) -> date:
    if not isinstance(value, str):
        raise TypeError("date must be an ISO string")
    return date.fromisoformat(value)


def _require_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("value must be a list")
    return value


def _validate_holiday_calendar(calendar: HolidayCalendar) -> None:
    if calendar.coverage_start > calendar.coverage_end:
        raise ValueError("holiday calendar coverage is reversed")
    if tuple(sorted(calendar.public_holidays)) != calendar.public_holidays:
        raise ValueError("holiday calendar dates must be sorted")
    if len(set(calendar.public_holidays)) != len(calendar.public_holidays):
        raise ValueError("holiday calendar dates must be unique")
    if any(not calendar.contains(value) for value in calendar.public_holidays):
        raise ValueError("holiday calendar date lies outside coverage")


def _validate_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _LOWERCASE_SHA256.fullmatch(value) is None:
        raise LightGBMArtifactError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_plain_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise LightGBMArtifactError(f"model_config.{field} must be an integer")
    return value


def _validate_model_config_values(value: object) -> dict[str, int | float]:
    if not isinstance(value, dict) or set(value) != _MODEL_CONFIG_KEYS:
        raise LightGBMArtifactError("model_config has an invalid field set")
    n_estimators = _require_plain_int(value["n_estimators"], field="n_estimators")
    num_leaves = _require_plain_int(value["num_leaves"], field="num_leaves")
    min_child_samples = _require_plain_int(value["min_child_samples"], field="min_child_samples")
    max_depth = _require_plain_int(value["max_depth"], field="max_depth")
    random_state = _require_plain_int(value["random_state"], field="random_state")
    learning_rate_value = value["learning_rate"]
    if isinstance(learning_rate_value, bool) or not isinstance(learning_rate_value, (int, float)):
        raise LightGBMArtifactError("model_config.learning_rate must be numeric")
    learning_rate = float(learning_rate_value)
    if n_estimators <= 0 or num_leaves <= 0 or min_child_samples <= 0:
        raise LightGBMArtifactError("model_config positive integers must be greater than zero")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise LightGBMArtifactError("model_config.learning_rate must be finite and positive")
    if max_depth < -1:
        raise LightGBMArtifactError("model_config.max_depth must not be below -1")
    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_child_samples": min_child_samples,
        "random_state": random_state,
    }


def _model_config_fields(config: LightGBMModelConfig) -> dict[str, int | float]:
    return _validate_model_config_values(
        {
            "n_estimators": config.n_estimators,
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "max_depth": config.max_depth,
            "min_child_samples": config.min_child_samples,
            "random_state": config.random_state,
        }
    )


def _parse_aware_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise LightGBMArtifactError(f"{field} must be an offset-bearing ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LightGBMArtifactError(f"{field} must be an offset-bearing ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LightGBMArtifactError(f"{field} must be an offset-bearing ISO timestamp")
    return parsed.astimezone(UTC)


def _validate_aware_timestamp_series(values: pd.Series) -> datetime:
    if values.empty:
        raise LightGBMArtifactError("forecast_origin_at has no eligible values")
    normalized: list[pd.Timestamp] = []
    for value in values:
        if pd.isna(value):
            raise LightGBMArtifactError("forecast_origin_at must be timezone-aware")
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise LightGBMArtifactError("forecast_origin_at must be timezone-aware") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise LightGBMArtifactError("forecast_origin_at must be timezone-aware")
        normalized.append(timestamp.tz_convert("UTC"))
    return max(normalized).to_pydatetime()


def _is_true_boolean(value: object) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _validate_weather_contract(training_rows: pd.DataFrame) -> None:
    for value_column, marker_column in _WEATHER_CONTRACT:
        if not training_rows[value_column].isna().all():
            raise LightGBMArtifactError("eligible training rows contain observed weather values")
        if not training_rows[marker_column].map(_is_true_boolean).all():
            raise LightGBMArtifactError(
                "eligible training rows must mark all weather values as missing"
            )


def _validate_fitted_model_contract(
    model: FittedLightGBMModel,
    *,
    manifest: LightGBMArtifactManifest,
) -> None:
    expected_columns = DEFAULT_RIDGE_FEATURE_SPEC.feature_columns
    if manifest.feature_columns != expected_columns:
        raise LightGBMArtifactError("manifest feature_columns do not match the supported spec")
    if not isinstance(model.pipeline, Pipeline):
        raise LightGBMArtifactError("fitted model pipeline has an unsupported type")
    if tuple(name for name, _ in model.pipeline.steps) != ("features", "model"):
        raise LightGBMArtifactError("fitted model pipeline has unsupported steps")
    features = model.pipeline.named_steps["features"]
    regressor = model.pipeline.named_steps["model"]
    if not isinstance(features, ColumnTransformer):
        raise LightGBMArtifactError("fitted model pipeline features step has an unsupported type")
    if not isinstance(regressor, LGBMRegressor):
        raise LightGBMArtifactError("fitted model pipeline model step has an unsupported type")
    try:
        check_is_fitted(model.pipeline)
        check_is_fitted(features)
        check_is_fitted(regressor)
    except NotFittedError as exc:
        raise LightGBMArtifactError("fitted model pipeline is not fitted") from exc
    if tuple(model.pipeline.feature_names_in_) != expected_columns:
        raise LightGBMArtifactError("fitted model pipeline feature input is unsupported")
    if tuple(features.feature_names_in_) != expected_columns:
        raise LightGBMArtifactError("fitted model preprocessing feature input is unsupported")
    if not isinstance(model.config, LightGBMModelConfig):
        raise LightGBMArtifactError("fitted model config has an unsupported type")
    if model.feature_columns != expected_columns:
        raise LightGBMArtifactError("fitted model feature_columns do not match the manifest")
    if model.config.feature_spec != DEFAULT_RIDGE_FEATURE_SPEC:
        raise LightGBMArtifactError("fitted model feature spec is unsupported")
    if model.config.feature_spec.feature_columns != expected_columns:
        raise LightGBMArtifactError("fitted model feature source is inconsistent")
    if _model_config_fields(model.config) != manifest.model_config:
        raise LightGBMArtifactError("fitted model config does not match model_config")
    if model.training_row_count != manifest.training_row_count:
        raise LightGBMArtifactError("fitted model training row count does not match manifest")


def _manifest_to_json(manifest: LightGBMArtifactManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "model_sha256": manifest.model_sha256,
        "training_data_sha256": manifest.training_data_sha256,
        "created_at": manifest.created_at.isoformat(),
        "trained_through_at": manifest.trained_through_at.isoformat(),
        "training_row_count": manifest.training_row_count,
        "feature_timezone": manifest.feature_timezone,
        "feature_columns": list(manifest.feature_columns),
        "model_config": manifest.model_config,
        **manifest.holiday_calendar.to_manifest_fields(),
        "evaluation_summary_path": manifest.evaluation_summary_path,
    }


def _manifest_from_json(payload: object) -> LightGBMArtifactManifest:
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_KEYS:
        raise LightGBMArtifactError("manifest has an invalid field set")
    if type(payload["schema_version"]) is not int or (
        payload["schema_version"] != ARTIFACT_SCHEMA_VERSION
    ):
        raise LightGBMArtifactError("unsupported artifact schema_version")
    if payload["model_name"] != ARTIFACT_MODEL_NAME:
        raise LightGBMArtifactError("unsupported artifact model_name")
    model_sha256 = _validate_sha256(payload["model_sha256"], field="model_sha256")
    training_data_sha256 = _validate_sha256(
        payload["training_data_sha256"], field="training_data_sha256"
    )
    expected_version = f"{ARTIFACT_MODEL_NAME}-{model_sha256[:12]}"
    if payload["model_version"] != expected_version:
        raise LightGBMArtifactError("model_version does not match model_sha256")
    training_row_count = payload["training_row_count"]
    if type(training_row_count) is not int or training_row_count <= 0:
        raise LightGBMArtifactError("training_row_count must be a positive integer")
    if payload["feature_timezone"] != FEATURE_TIMEZONE:
        raise LightGBMArtifactError("unsupported feature_timezone")
    raw_columns = payload["feature_columns"]
    if not isinstance(raw_columns, list) or not all(isinstance(item, str) for item in raw_columns):
        raise LightGBMArtifactError("feature_columns must be a list of strings")
    feature_columns = tuple(raw_columns)
    if feature_columns != DEFAULT_RIDGE_FEATURE_SPEC.feature_columns:
        raise LightGBMArtifactError("feature_columns do not match the supported spec")
    model_config = _validate_model_config_values(payload["model_config"])
    try:
        holiday_calendar = HolidayCalendar(
            coverage_start=_parse_iso_date(payload["holiday_calendar_start"]),
            coverage_end=_parse_iso_date(payload["holiday_calendar_end"]),
            public_holidays=tuple(
                _parse_iso_date(value) for value in _require_list(payload["public_holidays"])
            ),
        )
        _validate_holiday_calendar(holiday_calendar)
    except (TypeError, ValueError) as exc:
        raise LightGBMArtifactError("manifest holiday calendar is invalid") from exc
    evaluation_summary_path = payload["evaluation_summary_path"]
    if evaluation_summary_path is not None and not isinstance(evaluation_summary_path, str):
        raise LightGBMArtifactError("evaluation_summary_path must be a string or null")
    return LightGBMArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        model_name=ARTIFACT_MODEL_NAME,
        model_version=expected_version,
        model_sha256=model_sha256,
        training_data_sha256=training_data_sha256,
        created_at=_parse_aware_datetime(payload["created_at"], field="created_at"),
        trained_through_at=_parse_aware_datetime(
            payload["trained_through_at"], field="trained_through_at"
        ),
        training_row_count=training_row_count,
        feature_timezone=FEATURE_TIMEZONE,
        feature_columns=feature_columns,
        model_config=model_config,
        holiday_calendar=holiday_calendar,
        evaluation_summary_path=evaluation_summary_path,
    )


def export_lightgbm_artifact(
    supervised_frame: pd.DataFrame,
    *,
    source_csv_sha256: str,
    output_directory: str | Path,
    holiday_calendar: HolidayCalendar,
    model_config: LightGBMModelConfig,
    evaluation_summary_path: str | None = None,
) -> LightGBMArtifactManifest:
    output_path = _validated_local_path(output_directory)
    if os.path.lexists(output_path):
        raise LightGBMArtifactError(f"artifact destination already exists: {output_path}")
    _validate_sha256(source_csv_sha256, field="training_data_sha256")
    if evaluation_summary_path is not None and not isinstance(evaluation_summary_path, str):
        raise LightGBMArtifactError("evaluation_summary_path must be a string or null")
    try:
        _validate_holiday_calendar(holiday_calendar)
    except (TypeError, ValueError) as exc:
        raise LightGBMArtifactError("holiday calendar is invalid") from exc
    try:
        if model_config.feature_spec != DEFAULT_RIDGE_FEATURE_SPEC:
            raise LightGBMArtifactError("only the default feature spec is supported")
        model_config_fields = _model_config_fields(model_config)
        training_rows = select_training_rows(
            supervised_frame,
            feature_spec=DEFAULT_RIDGE_FEATURE_SPEC,
        )
    except ModelTrainingError as exc:
        raise LightGBMArtifactError(str(exc)) from exc
    _validate_weather_contract(training_rows)
    if "forecast_origin_at" not in training_rows:
        raise LightGBMArtifactError("forecast_origin_at is required for artifact training")
    trained_through_at = _validate_aware_timestamp_series(training_rows["forecast_origin_at"])
    try:
        fitted_model = fit_lightgbm_model(training_rows, config=model_config)
    except ModelTrainingError as exc:
        raise LightGBMArtifactError(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise LightGBMArtifactError("training data contains invalid numeric values") from exc
    provisional_manifest = LightGBMArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        model_name=ARTIFACT_MODEL_NAME,
        model_version="lightgbm-000000000000",
        model_sha256="0" * 64,
        training_data_sha256=source_csv_sha256,
        created_at=datetime.now(UTC),
        trained_through_at=trained_through_at,
        training_row_count=fitted_model.training_row_count,
        feature_timezone=FEATURE_TIMEZONE,
        feature_columns=fitted_model.feature_columns,
        model_config=model_config_fields,
        holiday_calendar=holiday_calendar,
        evaluation_summary_path=evaluation_summary_path,
    )
    if provisional_manifest.feature_columns != DEFAULT_RIDGE_FEATURE_SPEC.feature_columns:
        raise LightGBMArtifactError("fitted model feature_columns are unsupported")

    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{output_path.name}-", dir=output_path.parent)
        )
        model_path = temporary_path / ARTIFACT_MODEL_FILE_NAME
        joblib.dump(fitted_model, model_path)
        model_sha256 = sha256_file(model_path)
        manifest = replace(
            provisional_manifest,
            model_version=f"{ARTIFACT_MODEL_NAME}-{model_sha256[:12]}",
            model_sha256=model_sha256,
        )
        (temporary_path / ARTIFACT_MANIFEST_FILE_NAME).write_text(
            json.dumps(_manifest_to_json(manifest), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        load_lightgbm_artifact(temporary_path)
        temporary_path.rename(output_path)
        temporary_path = None
        return manifest
    except LightGBMArtifactError as exc:
        raise LightGBMArtifactSerializationError("could not serialize valid artifact") from exc
    except Exception as exc:
        raise LightGBMArtifactSerializationError(
            f"could not write artifact: {output_path}"
        ) from exc
    finally:
        if temporary_path is not None:
            shutil.rmtree(temporary_path, ignore_errors=True)


def load_lightgbm_artifact(path: str | Path) -> LoadedLightGBMArtifact:
    artifact_path = _validated_local_path(path)
    if not artifact_path.is_dir():
        raise LightGBMArtifactError(f"artifact directory does not exist: {artifact_path}")
    try:
        child_names = {child.name for child in artifact_path.iterdir()}
    except OSError as exc:
        raise LightGBMArtifactError(
            f"could not inspect artifact directory: {artifact_path}"
        ) from exc
    if child_names != _EXPECTED_BUNDLE_FILES:
        raise LightGBMArtifactError("artifact directory must contain exactly two bundle files")
    manifest_path = artifact_path / ARTIFACT_MANIFEST_FILE_NAME
    try:
        payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LightGBMArtifactError("artifact manifest is unreadable") from exc
    manifest = _manifest_from_json(payload)
    model_path = artifact_path / ARTIFACT_MODEL_FILE_NAME
    try:
        actual_model_sha256 = sha256_file(model_path)
    except OSError as exc:
        raise LightGBMArtifactError("artifact model is unreadable") from exc
    if actual_model_sha256 != manifest.model_sha256:
        raise LightGBMArtifactError("artifact model checksum does not match manifest")
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise LightGBMArtifactError("artifact model could not be deserialized") from exc
    if not isinstance(model, FittedLightGBMModel):
        raise LightGBMArtifactError("artifact model has an unsupported type")
    try:
        _validate_fitted_model_contract(model, manifest=manifest)
    except LightGBMArtifactError:
        raise
    except Exception as exc:
        raise LightGBMArtifactError("artifact model contract is invalid") from exc
    return LoadedLightGBMArtifact(manifest=manifest, model=model)
