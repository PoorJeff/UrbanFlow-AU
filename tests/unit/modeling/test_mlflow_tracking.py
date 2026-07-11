from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from urbanflow.modeling.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    MLflowTrackingConfig,
    MLflowTrackingError,
    final_test_metric_values,
    load_evaluation_summary,
    normalize_model_name,
    tracking_params_from_summary,
    tracking_tags_for_model,
    validate_evaluation_summary,
    validation_window_metric_steps,
)


def lightgbm_summary_path() -> Path:
    return (
        Path(__file__).parents[3]
        / "docs"
        / "examples"
        / "modeling"
        / "lightgbm_evaluation_summary.json"
    )


def ridge_summary_path() -> Path:
    return (
        Path(__file__).parents[3]
        / "docs"
        / "examples"
        / "modeling"
        / "ridge_evaluation_summary.json"
    )


def load_lightgbm_summary() -> dict[str, object]:
    return json.loads(lightgbm_summary_path().read_text(encoding="utf-8"))


def load_ridge_summary() -> dict[str, object]:
    return json.loads(ridge_summary_path().read_text(encoding="utf-8"))


def test_tracking_config_defaults_to_local_baseline_experiment() -> None:
    config = MLflowTrackingConfig()

    assert config.experiment_name == DEFAULT_EXPERIMENT_NAME
    assert config.tracking_uri is None
    assert config.extra_tags == {}


def test_load_evaluation_summary_reads_json_object() -> None:
    summary = load_evaluation_summary(lightgbm_summary_path())

    assert summary["input_path"] == "docs/examples/modeling/synthetic_supervised_rows.csv"
    assert summary["validation_window_count"] == 1


def test_load_evaluation_summary_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(MLflowTrackingError, match="summary JSON does not exist"):
        load_evaluation_summary(tmp_path / "missing.json")


def test_load_evaluation_summary_rejects_non_object_json(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(MLflowTrackingError, match="summary JSON must contain an object"):
        load_evaluation_summary(path)


def test_validate_evaluation_summary_rejects_missing_required_field() -> None:
    summary = load_lightgbm_summary()
    del summary["final_test"]

    with pytest.raises(
        MLflowTrackingError,
        match="missing required evaluation summary field: final_test",
    ):
        validate_evaluation_summary(summary)


def test_validate_evaluation_summary_rejects_invalid_window_shape() -> None:
    summary = load_lightgbm_summary()
    validation_windows = summary["validation_windows"]
    assert isinstance(validation_windows, list)
    validation_windows[0] = "not-a-window"

    with pytest.raises(
        MLflowTrackingError,
        match="summary field must be an object: validation_windows.0",
    ):
        validate_evaluation_summary(summary)


def test_normalize_model_name_accepts_supported_models() -> None:
    assert normalize_model_name("Ridge") == "ridge"
    assert normalize_model_name("lightgbm") == "lightgbm"


def test_normalize_model_name_rejects_unknown_model() -> None:
    with pytest.raises(MLflowTrackingError, match="unsupported model name"):
        normalize_model_name("xgboost")


def test_tracking_tags_for_model_merges_defaults_and_extra_tags() -> None:
    tags = tracking_tags_for_model("lightgbm", extra_tags={"owner": "urbanflow"})

    assert tags == {
        "owner": "urbanflow",
        "urbanflow.model": "lightgbm",
        "urbanflow.source": "supervised_csv",
        "urbanflow.stage": "local_baseline",
        "urbanflow.summary_schema": "rolling_origin_v1",
    }


def test_tracking_params_from_summary_returns_run_descriptors() -> None:
    params = tracking_params_from_summary(load_lightgbm_summary(), model_name="lightgbm")

    assert params == {
        "final_test_window": "final_test_2025-02",
        "input_path": "docs/examples/modeling/synthetic_supervised_rows.csv",
        "model_name": "lightgbm",
        "row_count": 1464,
        "validation_window_count": 1,
    }


def test_final_test_metric_values_flattens_model_and_baseline_metrics() -> None:
    metrics = final_test_metric_values(load_lightgbm_summary())

    assert metrics == {
        "final_test_mae": 0.95,
        "final_test_relative_wape_improvement": 0.45,
        "final_test_rmse": 1.45,
        "final_test_row_count": 672,
        "final_test_seasonal_naive_mae": 1.8,
        "final_test_seasonal_naive_rmse": 2.3,
        "final_test_seasonal_naive_wape": 0.1,
        "final_test_wape": 0.055,
    }


def test_final_test_metric_values_skips_non_numeric_optional_values() -> None:
    summary = load_lightgbm_summary()
    final_test = summary["final_test"]
    assert isinstance(final_test, dict)
    comparison = final_test["model_comparison"]
    assert isinstance(comparison, dict)
    comparison["relative_wape_improvement"] = None
    overall = final_test["overall"]
    assert isinstance(overall, dict)
    overall["rmse"] = float("nan")

    metrics = final_test_metric_values(summary)

    assert "final_test_relative_wape_improvement" not in metrics
    assert "final_test_rmse" not in metrics
    assert math.isclose(metrics["final_test_wape"], 0.055)


def test_validation_window_metric_steps_flattens_each_window_with_step() -> None:
    steps = validation_window_metric_steps(load_ridge_summary())

    assert steps == [
        (
            0,
            {
                "validation_mae": 1.2345,
                "validation_relative_wape_improvement": 0.1268817204301075,
                "validation_rmse": 1.7654,
                "validation_seasonal_naive_wape": 0.093,
                "validation_wape": 0.0812,
            },
        )
    ]


def test_validation_window_metric_steps_skips_missing_comparison_metric() -> None:
    summary = load_ridge_summary()
    summary = deepcopy(summary)
    validation_windows = summary["validation_windows"]
    assert isinstance(validation_windows, list)
    window = validation_windows[0]
    assert isinstance(window, dict)
    comparison = window["model_comparison"]
    assert isinstance(comparison, dict)
    comparison["relative_wape_improvement"] = None

    steps = validation_window_metric_steps(summary)

    assert "validation_relative_wape_improvement" not in steps[0][1]
    assert steps[0][1]["validation_wape"] == 0.0812
