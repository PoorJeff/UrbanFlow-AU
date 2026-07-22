from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from urbanflow.modeling.feature_matrix import DEFAULT_RIDGE_FEATURE_SPEC
from urbanflow.modeling.lightgbm import FittedLightGBMModel, LightGBMModelConfig
from urbanflow.modeling.lightgbm_artifact import (
    HolidayCalendar,
    LightGBMArtifactError,
    export_lightgbm_artifact,
    load_lightgbm_artifact,
)
from urbanflow.modeling.supervised_csv import read_supervised_csv, sha256_file


def supervised_rows() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-04-05 00:00",
        periods=192,
        freq="h",
        tz="Australia/Melbourne",
    )
    index = pd.Series(range(len(timestamps)), dtype="float64")
    frame = pd.DataFrame(
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
            "is_weekend": [timestamp.weekday() >= 5 for timestamp in timestamps],
            "is_public_holiday": [False] * len(timestamps),
            "hour_sin": [math.sin((timestamp.hour / 24.0) * math.tau) for timestamp in timestamps],
            "hour_cos": [math.cos((timestamp.hour / 24.0) * math.tau) for timestamp in timestamps],
            "weekday_sin": [
                math.sin((timestamp.weekday() / 7.0) * math.tau) for timestamp in timestamps
            ],
            "weekday_cos": [
                math.cos((timestamp.weekday() / 7.0) * math.tau) for timestamp in timestamps
            ],
            "temperature": [pd.NA] * len(timestamps),
            "temperature_missing": [True] * len(timestamps),
            "rainfall": [pd.NA] * len(timestamps),
            "rainfall_missing": [True] * len(timestamps),
            "wind_speed": [pd.NA] * len(timestamps),
            "wind_speed_missing": [True] * len(timestamps),
        }
    )
    return frame


def write_supervised_csv(path: Path) -> Path:
    supervised_rows().to_csv(path, index=False)
    return path


def write_holiday_calendar(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "coverage_start": "2025-01-01",
        "coverage_end": "2026-12-31",
        "public_holidays": ["2025-01-27", "2026-01-26"],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def build_artifact(tmp_path: Path, *, name: str = "artifact") -> Path:
    csv_path = write_supervised_csv(tmp_path / f"{name}.csv")
    calendar = HolidayCalendar.from_json_file(
        write_holiday_calendar(tmp_path / f"{name}-holidays.json")
    )
    output = tmp_path / name
    export_lightgbm_artifact(
        read_supervised_csv(csv_path),
        source_csv_sha256=sha256_file(csv_path),
        output_directory=output,
        holiday_calendar=calendar,
        model_config=LightGBMModelConfig(n_estimators=5, min_child_samples=1),
    )
    return output


def mutate_manifest(artifact: Path, **updates: object) -> None:
    path = artifact / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def refresh_model_integrity(artifact: Path) -> None:
    model_hash = sha256_file(artifact / "model.joblib")
    mutate_manifest(
        artifact,
        model_sha256=model_hash,
        model_version=f"lightgbm-{model_hash[:12]}",
    )


def test_artifact_round_trip_preserves_pipeline_and_manifest(tmp_path: Path) -> None:
    csv_path = write_supervised_csv(tmp_path / "rows.csv")
    frame = read_supervised_csv(csv_path)
    calendar = HolidayCalendar.from_json_file(write_holiday_calendar(tmp_path / "holidays.json"))

    manifest = export_lightgbm_artifact(
        frame,
        source_csv_sha256=sha256_file(csv_path),
        output_directory=tmp_path / "artifact",
        holiday_calendar=calendar,
        model_config=LightGBMModelConfig(n_estimators=5, min_child_samples=1),
    )
    loaded = load_lightgbm_artifact(tmp_path / "artifact")

    assert manifest.model_name == "lightgbm"
    assert manifest.model_version == f"lightgbm-{manifest.model_sha256[:12]}"
    assert manifest.feature_columns == DEFAULT_RIDGE_FEATURE_SPEC.feature_columns
    assert manifest.training_row_count == 192
    assert manifest.trained_through_at == frame["forecast_origin_at"].max()
    assert loaded.manifest == manifest
    assert isinstance(loaded.model, FittedLightGBMModel)
    assert {path.name for path in (tmp_path / "artifact").iterdir()} == {
        "manifest.json",
        "model.joblib",
    }


def test_export_creates_missing_parent_and_refuses_existing_destination(tmp_path: Path) -> None:
    csv_path = write_supervised_csv(tmp_path / "rows.csv")
    frame = read_supervised_csv(csv_path)
    calendar = HolidayCalendar.from_json_file(write_holiday_calendar(tmp_path / "holidays.json"))
    output = tmp_path / "missing" / "parent" / "artifact"

    export_lightgbm_artifact(
        frame,
        source_csv_sha256=sha256_file(csv_path),
        output_directory=output,
        holiday_calendar=calendar,
        model_config=LightGBMModelConfig(n_estimators=5, min_child_samples=1),
    )
    with pytest.raises(LightGBMArtifactError, match="already exists"):
        export_lightgbm_artifact(
            frame,
            source_csv_sha256=sha256_file(csv_path),
            output_directory=output,
            holiday_calendar=calendar,
            model_config=LightGBMModelConfig(n_estimators=5, min_child_samples=1),
        )


@pytest.mark.parametrize(
    "output",
    ["s3://bucket/artifact", Path("s3://bucket/artifact"), "s3:bucket/artifact"],
)
def test_export_rejects_remote_looking_path_before_writing(
    tmp_path: Path,
    output: str | Path,
) -> None:
    csv_path = write_supervised_csv(tmp_path / "rows.csv")
    calendar = HolidayCalendar.from_json_file(write_holiday_calendar(tmp_path / "holidays.json"))

    with pytest.raises(LightGBMArtifactError, match="local path"):
        export_lightgbm_artifact(
            read_supervised_csv(csv_path),
            source_csv_sha256=sha256_file(csv_path),
            output_directory=output,
            holiday_calendar=calendar,
            model_config=LightGBMModelConfig(n_estimators=5, min_child_samples=1),
        )


@pytest.mark.parametrize(
    "artifact_path",
    ["s3://bucket/artifact", Path("s3://bucket/artifact"), "s3:bucket/artifact"],
)
def test_loader_rejects_all_multicharacter_uri_schemes(artifact_path: str | Path) -> None:
    with pytest.raises(LightGBMArtifactError, match="local path"):
        load_lightgbm_artifact(artifact_path)


def test_loader_accepts_windows_drive_prefix_before_directory_check(monkeypatch) -> None:
    monkeypatch.setattr(Path, "is_dir", lambda self: False)

    with pytest.raises(LightGBMArtifactError, match="artifact directory does not exist"):
        load_lightgbm_artifact(Path(r"C:\models\artifact"))


@pytest.mark.parametrize(
    ("weather_column", "weather_value"),
    [("temperature", 20.0), ("rainfall_missing", False)],
)
def test_export_rejects_weather_incompatible_eligible_rows(
    tmp_path: Path,
    weather_column: str,
    weather_value: object,
) -> None:
    frame = supervised_rows()
    frame.loc[0, weather_column] = weather_value
    calendar = HolidayCalendar.from_json_file(write_holiday_calendar(tmp_path / "holidays.json"))

    with pytest.raises(LightGBMArtifactError, match="weather"):
        export_lightgbm_artifact(
            frame,
            source_csv_sha256="0" * 64,
            output_directory=tmp_path / "artifact",
            holiday_calendar=calendar,
            model_config=LightGBMModelConfig(n_estimators=5, min_child_samples=1),
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"coverage_start": "2025-01-01", "coverage_end": "2025-12-31"},
        {
            "coverage_start": "not-a-date",
            "coverage_end": "2025-12-31",
            "public_holidays": [],
        },
        {
            "coverage_start": "2025-12-31",
            "coverage_end": "2025-01-01",
            "public_holidays": [],
        },
        {
            "coverage_start": "2025-01-01",
            "coverage_end": "2025-12-31",
            "public_holidays": ["2025-01-27", "2025-01-27"],
        },
        {
            "coverage_start": "2025-01-01",
            "coverage_end": "2025-12-31",
            "public_holidays": ["2025-06-01", "2025-01-27"],
        },
        {
            "coverage_start": "2025-01-01",
            "coverage_end": "2025-12-31",
            "public_holidays": ["2026-01-26"],
        },
    ],
)
def test_holiday_calendar_rejects_invalid_contract(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    path = tmp_path / "holidays.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LightGBMArtifactError, match="holiday calendar"):
        HolidayCalendar.from_json_file(path)


def test_holiday_calendar_contains_only_covered_dates(tmp_path: Path) -> None:
    calendar = HolidayCalendar.from_json_file(write_holiday_calendar(tmp_path / "holidays.json"))

    assert calendar.contains(date(2025, 1, 1))
    assert calendar.contains(date(2026, 12, 31))
    assert not calendar.contains(date(2027, 1, 1))


def test_loader_rejects_missing_and_extra_bundle_members(tmp_path: Path) -> None:
    missing = build_artifact(tmp_path, name="missing")
    (missing / "model.joblib").unlink()
    with pytest.raises(LightGBMArtifactError, match="exactly"):
        load_lightgbm_artifact(missing)

    extra = build_artifact(tmp_path, name="extra")
    (extra / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(LightGBMArtifactError, match="exactly"):
        load_lightgbm_artifact(extra)


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": 2},
        {"model_name": "ridge"},
        {"model_sha256": "A" * 64},
        {"model_sha256": "0" * 64},
        {"model_version": "lightgbm-wrong"},
        {"created_at": "2025-01-01T00:00:00"},
        {"trained_through_at": "2025-01-01T00:00:00"},
        {"training_data_sha256": "A" * 64},
    ],
)
def test_loader_rejects_invalid_manifest_scalar_contract(
    tmp_path: Path,
    update: dict[str, object],
) -> None:
    artifact = build_artifact(tmp_path)
    mutate_manifest(artifact, **update)

    with pytest.raises(LightGBMArtifactError):
        load_lightgbm_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_estimators", True),
        ("n_estimators", 0),
        ("learning_rate", 0.0),
        ("learning_rate", float("inf")),
        ("max_depth", -2),
        ("random_state", 1.5),
    ],
)
def test_loader_rejects_invalid_model_config(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    artifact = build_artifact(tmp_path)
    manifest_path = artifact / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model_config"][field] = value
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(LightGBMArtifactError, match="model_config"):
        load_lightgbm_artifact(artifact)


def test_loader_rejects_manifest_feature_columns_outside_default_spec(tmp_path: Path) -> None:
    artifact = build_artifact(tmp_path)
    mutate_manifest(artifact, feature_columns=list(DEFAULT_RIDGE_FEATURE_SPEC.feature_columns[:-1]))

    with pytest.raises(LightGBMArtifactError, match="feature_columns"):
        load_lightgbm_artifact(artifact)


def test_loader_rejects_fitted_model_feature_columns_mismatch(tmp_path: Path) -> None:
    artifact = build_artifact(tmp_path)
    model_path = artifact / "model.joblib"
    model = joblib.load(model_path)
    joblib.dump(replace(model, feature_columns=model.feature_columns[:-1]), model_path)
    refresh_model_integrity(artifact)

    with pytest.raises(LightGBMArtifactError, match="feature_columns"):
        load_lightgbm_artifact(artifact)


def test_loader_rejects_fitted_config_feature_source_mismatch(tmp_path: Path) -> None:
    artifact = build_artifact(tmp_path)
    model_path = artifact / "model.joblib"
    model = joblib.load(model_path)
    altered_spec = replace(
        DEFAULT_RIDGE_FEATURE_SPEC,
        numeric_columns=DEFAULT_RIDGE_FEATURE_SPEC.numeric_columns[:-1],
    )
    joblib.dump(replace(model, config=replace(model.config, feature_spec=altered_spec)), model_path)
    refresh_model_integrity(artifact)

    with pytest.raises(LightGBMArtifactError, match="feature"):
        load_lightgbm_artifact(artifact)


@pytest.mark.parametrize(
    ("replacement", "error_match"),
    [
        ({"pipeline": None}, "pipeline"),
        ({"pipeline": Pipeline([("scale", StandardScaler())])}, "fitted"),
        ({"config": None}, "config"),
    ],
)
def test_loader_normalizes_invalid_nested_fitted_model_contract(
    tmp_path: Path,
    replacement: dict[str, object],
    error_match: str,
) -> None:
    artifact = build_artifact(tmp_path)
    model_path = artifact / "model.joblib"
    model = joblib.load(model_path)
    joblib.dump(replace(model, **replacement), model_path)
    refresh_model_integrity(artifact)

    with pytest.raises(LightGBMArtifactError, match=error_match):
        load_lightgbm_artifact(artifact)


def test_loader_rejects_fitted_pipeline_outside_project_structure(tmp_path: Path) -> None:
    artifact = build_artifact(tmp_path)
    model_path = artifact / "model.joblib"
    model = joblib.load(model_path)
    wrong_pipeline = Pipeline([("scale", StandardScaler())]).fit(
        pd.DataFrame({"value": [1.0, 2.0]})
    )
    joblib.dump(replace(model, pipeline=wrong_pipeline), model_path)
    refresh_model_integrity(artifact)

    with pytest.raises(LightGBMArtifactError, match="pipeline"):
        load_lightgbm_artifact(artifact)
