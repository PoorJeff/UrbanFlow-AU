from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_EXPERIMENT_NAME = "urbanflow-local-baselines"
SUPPORTED_MODEL_NAMES = frozenset({"ridge", "lightgbm"})


class MLflowTrackingError(ValueError):
    """Raised when a local evaluation summary cannot be tracked."""


class MLflowTrackingAdapter(Protocol):
    """Boundary for MLflow operations so tests can use a fake logger."""

    def set_tracking_uri(self, tracking_uri: str) -> None:
        """Configure the backing MLflow tracking store."""

    def set_experiment(self, experiment_name: str) -> None:
        """Select or create an MLflow experiment."""

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Log run tags."""

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log run parameters."""

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        """Log one scalar metric."""

    def log_artifact(self, local_path: Path, *, artifact_path: str | None = None) -> None:
        """Log one local artifact file."""


@dataclass(frozen=True)
class MLflowTrackingConfig:
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    tracking_uri: str | None = None
    extra_tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MLflowRunResult:
    run_id: str
    experiment_id: str
    tracking_uri: str


def _field_path(parent: str, key: str) -> str:
    return key if not parent else f"{parent}.{key}"


def _required(mapping: Mapping[str, Any], key: str, *, path: str = "") -> Any:
    if key not in mapping:
        raise MLflowTrackingError(
            f"missing required evaluation summary field: {_field_path(path, key)}"
        )
    return mapping[key]


def _required_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MLflowTrackingError(f"summary field must be an object: {path}")
    return value


def _required_sequence(value: Any, *, path: str) -> list[Any]:
    if isinstance(value, str) or not isinstance(value, list):
        raise MLflowTrackingError(f"summary field must be a list: {path}")
    return value


def _numeric_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(metric) or math.isinf(metric):
        return None
    return metric


def _add_metric(metrics: dict[str, float], name: str, value: Any) -> None:
    metric = _numeric_metric(value)
    if metric is not None:
        metrics[name] = metric


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized not in SUPPORTED_MODEL_NAMES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_NAMES))
        raise MLflowTrackingError(
            f"unsupported model name: {model_name}. Supported models: {supported}"
        )
    return normalized


def load_evaluation_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MLflowTrackingError(f"summary JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MLflowTrackingError(f"could not read summary JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MLflowTrackingError(f"invalid summary JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise MLflowTrackingError("summary JSON must contain an object")
    validate_evaluation_summary(payload)
    return payload


def _validate_metric_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    metrics = _required_mapping(value, path=path)
    for field_name in ("row_count", "mae", "rmse", "wape"):
        _required(metrics, field_name, path=path)
    return metrics


def _validate_window(value: Any, *, path: str) -> Mapping[str, Any]:
    window = _required_mapping(value, path=path)
    for field_name in (
        "name",
        "start",
        "end",
        "train_end",
        "training_row_count",
        "overall",
        "horizon_metrics",
    ):
        _required(window, field_name, path=path)
    _validate_metric_mapping(window["overall"], path=_field_path(path, "overall"))

    if "seasonal_naive_overall" in window:
        _validate_metric_mapping(
            window["seasonal_naive_overall"],
            path=_field_path(path, "seasonal_naive_overall"),
        )
    if "model_comparison" in window:
        _required_mapping(
            window["model_comparison"],
            path=_field_path(path, "model_comparison"),
        )
    return window


def validate_evaluation_summary(summary: Mapping[str, Any]) -> None:
    for field_name in (
        "input_path",
        "row_count",
        "validation_window_count",
        "validation_windows",
        "final_test",
    ):
        _required(summary, field_name)

    validation_windows = _required_sequence(
        summary["validation_windows"],
        path="validation_windows",
    )
    for index, window in enumerate(validation_windows):
        _validate_window(window, path=f"validation_windows.{index}")
    _validate_window(summary["final_test"], path="final_test")


def tracking_tags_for_model(
    model_name: str,
    *,
    extra_tags: Mapping[str, str] | None = None,
) -> dict[str, str]:
    tags = dict(extra_tags or {})
    tags.update(
        {
            "urbanflow.model": normalize_model_name(model_name),
            "urbanflow.source": "supervised_csv",
            "urbanflow.stage": "local_baseline",
            "urbanflow.summary_schema": "rolling_origin_v1",
        }
    )
    return tags


def tracking_params_from_summary(
    summary: Mapping[str, Any],
    *,
    model_name: str,
) -> dict[str, object]:
    validate_evaluation_summary(summary)
    final_test = _required_mapping(summary["final_test"], path="final_test")
    return {
        "final_test_window": final_test["name"],
        "input_path": summary["input_path"],
        "model_name": normalize_model_name(model_name),
        "row_count": summary["row_count"],
        "validation_window_count": summary["validation_window_count"],
    }


def final_test_metric_values(summary: Mapping[str, Any]) -> dict[str, float]:
    validate_evaluation_summary(summary)
    final_test = _required_mapping(summary["final_test"], path="final_test")
    overall = _validate_metric_mapping(final_test["overall"], path="final_test.overall")
    metrics: dict[str, float] = {}
    _add_metric(metrics, "final_test_row_count", overall["row_count"])
    _add_metric(metrics, "final_test_mae", overall["mae"])
    _add_metric(metrics, "final_test_rmse", overall["rmse"])
    _add_metric(metrics, "final_test_wape", overall["wape"])

    seasonal_naive_overall = final_test.get("seasonal_naive_overall")
    if seasonal_naive_overall is not None:
        baseline = _validate_metric_mapping(
            seasonal_naive_overall,
            path="final_test.seasonal_naive_overall",
        )
        _add_metric(metrics, "final_test_seasonal_naive_mae", baseline["mae"])
        _add_metric(metrics, "final_test_seasonal_naive_rmse", baseline["rmse"])
        _add_metric(metrics, "final_test_seasonal_naive_wape", baseline["wape"])

    comparison = final_test.get("model_comparison")
    if comparison is not None:
        comparison_mapping = _required_mapping(
            comparison,
            path="final_test.model_comparison",
        )
        _add_metric(
            metrics,
            "final_test_relative_wape_improvement",
            comparison_mapping.get("relative_wape_improvement"),
        )
    return metrics


def _validation_metrics_for_window(window: Mapping[str, Any]) -> dict[str, float]:
    overall = _validate_metric_mapping(window["overall"], path=f"{window['name']}.overall")
    metrics: dict[str, float] = {}
    _add_metric(metrics, "validation_mae", overall["mae"])
    _add_metric(metrics, "validation_rmse", overall["rmse"])
    _add_metric(metrics, "validation_wape", overall["wape"])

    seasonal_naive_overall = window.get("seasonal_naive_overall")
    if seasonal_naive_overall is not None:
        baseline = _validate_metric_mapping(
            seasonal_naive_overall,
            path=f"{window['name']}.seasonal_naive_overall",
        )
        _add_metric(metrics, "validation_seasonal_naive_wape", baseline["wape"])

    comparison = window.get("model_comparison")
    if comparison is not None:
        comparison_mapping = _required_mapping(
            comparison,
            path=f"{window['name']}.model_comparison",
        )
        _add_metric(
            metrics,
            "validation_relative_wape_improvement",
            comparison_mapping.get("relative_wape_improvement"),
        )
    return metrics


def validation_window_metric_steps(
    summary: Mapping[str, Any],
) -> list[tuple[int, dict[str, float]]]:
    validate_evaluation_summary(summary)
    validation_windows = _required_sequence(
        summary["validation_windows"],
        path="validation_windows",
    )
    return [
        (index, _validation_metrics_for_window(_required_mapping(window, path=str(index))))
        for index, window in enumerate(validation_windows)
    ]
