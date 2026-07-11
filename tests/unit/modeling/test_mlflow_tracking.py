from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException

from urbanflow.modeling.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    MLflowRunResult,
    MLflowTrackingConfig,
    MLflowTrackingError,
    final_test_metric_values,
    load_evaluation_summary,
    normalize_model_name,
    track_evaluation_summary,
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


def lightgbm_report_path() -> Path:
    return (
        Path(__file__).parents[3]
        / "docs"
        / "examples"
        / "modeling"
        / "lightgbm_evaluation_report.md"
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


@dataclass(frozen=True)
class FakeRunInfo:
    run_id: str = "run-123"
    experiment_id: str = "experiment-456"


@dataclass(frozen=True)
class FakeActiveRun:
    info: FakeRunInfo = FakeRunInfo()


class FakeRunContext:
    def __init__(self, adapter: FakeMLflowTrackingAdapter) -> None:
        self.adapter = adapter

    def __enter__(self) -> FakeActiveRun:
        self.adapter.started_runs += 1
        return FakeActiveRun()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.adapter.finished_runs += 1


class FakeMLflowTrackingAdapter:
    def __init__(self, *, tracking_uri: str = "file:///default-mlruns") -> None:
        self.tracking_uri = tracking_uri
        self.tracking_uris: list[str] = []
        self.experiments: list[str] = []
        self.tags: dict[str, str] = {}
        self.params: dict[str, object] = {}
        self.metrics: list[tuple[str, float, int | None]] = []
        self.artifacts: list[tuple[Path, str | None]] = []
        self.started_runs = 0
        self.finished_runs = 0

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uris.append(tracking_uri)
        self.tracking_uri = tracking_uri

    def get_tracking_uri(self) -> str:
        return self.tracking_uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiments.append(experiment_name)

    def start_run(self) -> FakeRunContext:
        return FakeRunContext(self)

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        self.metrics.append((key, value, step))

    def log_artifact(self, local_path: Path, *, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))


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


def test_track_evaluation_summary_logs_run_payload(tmp_path) -> None:
    report_path = tmp_path / "lightgbm_report.md"
    report_path.write_text("# LightGBM report\n", encoding="utf-8")
    adapter = FakeMLflowTrackingAdapter()
    config = MLflowTrackingConfig(
        tracking_uri="file:///tmp/mlruns",
        extra_tags={"owner": "urbanflow"},
    )

    result = track_evaluation_summary(
        "LightGBM",
        lightgbm_summary_path(),
        report_path=report_path,
        config=config,
        adapter=adapter,
    )

    assert result == MLflowRunResult(
        run_id="run-123",
        experiment_id="experiment-456",
        tracking_uri="file:///tmp/mlruns",
    )
    assert adapter.tracking_uris == ["file:///tmp/mlruns"]
    assert adapter.experiments == [DEFAULT_EXPERIMENT_NAME]
    assert adapter.started_runs == 1
    assert adapter.finished_runs == 1
    assert adapter.tags == {
        "owner": "urbanflow",
        "urbanflow.model": "lightgbm",
        "urbanflow.source": "supervised_csv",
        "urbanflow.stage": "local_baseline",
        "urbanflow.summary_schema": "rolling_origin_v1",
    }
    assert adapter.params == {
        "final_test_window": "final_test_2025-02",
        "input_path": "docs/examples/modeling/synthetic_supervised_rows.csv",
        "model_name": "lightgbm",
        "report_path": str(report_path),
        "row_count": 1464,
        "summary_json_path": str(lightgbm_summary_path()),
        "validation_window_count": 1,
    }
    assert ("final_test_wape", 0.055, None) in adapter.metrics
    assert ("final_test_seasonal_naive_wape", 0.1, None) in adapter.metrics
    assert ("validation_wape", 0.065, 0) in adapter.metrics
    assert ("validation_relative_wape_improvement", 0.2777777777777778, 0) in (adapter.metrics)
    assert adapter.artifacts == [
        (lightgbm_summary_path(), "evaluation"),
        (report_path, "reports"),
    ]


def test_track_evaluation_summary_uses_default_tracking_uri_without_setting_one() -> None:
    adapter = FakeMLflowTrackingAdapter(tracking_uri="file:///existing-mlruns")

    result = track_evaluation_summary(
        "ridge",
        ridge_summary_path(),
        adapter=adapter,
    )

    assert adapter.tracking_uris == []
    assert result.tracking_uri == "file:///existing-mlruns"
    assert adapter.artifacts == [(ridge_summary_path(), "evaluation")]


def test_track_evaluation_summary_rejects_missing_report_path(tmp_path) -> None:
    adapter = FakeMLflowTrackingAdapter()

    with pytest.raises(MLflowTrackingError, match="report Markdown does not exist"):
        track_evaluation_summary(
            "ridge",
            ridge_summary_path(),
            report_path=tmp_path / "missing.md",
            adapter=adapter,
        )

    assert adapter.started_runs == 0


def test_track_evaluation_summary_converts_mlflow_errors() -> None:
    class FailingMLflowTrackingAdapter(FakeMLflowTrackingAdapter):
        def log_params(self, params: dict[str, object]) -> None:
            raise MlflowException("tracking store unavailable")

    with pytest.raises(MLflowTrackingError, match="MLflow tracking failed"):
        track_evaluation_summary(
            "ridge",
            ridge_summary_path(),
            adapter=FailingMLflowTrackingAdapter(),
        )


def test_track_evaluation_summary_writes_file_backed_mlflow_run(
    tmp_path,
    monkeypatch,
) -> None:
    import mlflow

    working_dir = tmp_path / "working-directory"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_root = tmp_path / "mlflow-store"
    tracking_root.mkdir()
    tracking_uri = tracking_root.as_uri()
    original_tracking_uri = mlflow.get_tracking_uri()

    try:
        result = track_evaluation_summary(
            "lightgbm",
            lightgbm_summary_path(),
            report_path=lightgbm_report_path(),
            config=MLflowTrackingConfig(
                experiment_name="urbanflow-smoke-test",
                tracking_uri=tracking_uri,
            ),
        )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)

    assert result.run_id
    assert result.experiment_id
    assert result.tracking_uri == tracking_uri
    assert mlflow.active_run() is None
    logged_file_names = {path.name for path in tracking_root.rglob("*") if path.is_file()}
    assert "lightgbm_evaluation_summary.json" in logged_file_names
    assert "lightgbm_evaluation_report.md" in logged_file_names
    assert not (working_dir / "mlruns").exists()
