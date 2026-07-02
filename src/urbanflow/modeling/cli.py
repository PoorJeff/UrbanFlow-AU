from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from urbanflow.modeling.evaluation import (
    ModelWindowEvaluation,
    RollingOriginRidgeEvaluation,
    evaluate_rolling_origin_ridge,
)
from urbanflow.modeling.feature_matrix import ModelTrainingError
from urbanflow.modeling.metrics import RegressionMetrics
from urbanflow.modeling.ridge import RidgeModelConfig
from urbanflow.modeling.splits import SplitConfigError, build_rolling_origin_splits

TIMESTAMP_COLUMNS = ("forecast_origin_at", "target_observed_at")


class RidgeEvaluationCliError(ValueError):
    """Raised when local Ridge evaluation CLI input is invalid."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a local Ridge baseline from supervised feature rows."
    )
    parser.add_argument(
        "supervised_csv",
        type=Path,
        help="CSV containing already-built supervised feature rows.",
    )
    parser.add_argument(
        "--validation-months",
        type=int,
        default=3,
        help="Positive number of validation months before the final test month.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Positive Ridge regularization strength.",
    )
    return parser


def _positive_integer(value: int, *, name: str) -> int:
    if value <= 0:
        raise RidgeEvaluationCliError(f"{name} must be greater than zero")
    return value


def _positive_float(value: float, *, name: str) -> float:
    if value <= 0:
        raise RidgeEvaluationCliError(f"{name} must be greater than zero")
    return value


def _read_supervised_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RidgeEvaluationCliError(f"CSV file does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise RidgeEvaluationCliError(f"could not read supervised CSV: {path}") from exc

    for column in TIMESTAMP_COLUMNS:
        if column in frame.columns:
            try:
                frame[column] = pd.to_datetime(frame[column])
            except (TypeError, ValueError) as exc:
                raise RidgeEvaluationCliError(
                    f"could not parse timestamp column: {column}"
                ) from exc
    return frame


def _json_scalar(value: object) -> object:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _metrics_summary(metrics: RegressionMetrics) -> dict[str, object]:
    return {
        "row_count": metrics.row_count,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "wape": metrics.wape,
    }


def _horizon_metric_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        records.append({key: _json_scalar(value) for key, value in record.items()})
    return records


def _timestamp_text(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).isoformat()


def _window_summary(evaluation: ModelWindowEvaluation) -> dict[str, object]:
    window = evaluation.window
    return {
        "name": window.name,
        "start": _timestamp_text(window.start),
        "end": _timestamp_text(window.end),
        "train_end": _timestamp_text(window.train_end),
        "training_row_count": evaluation.model.training_row_count,
        "overall": _metrics_summary(evaluation.overall_metrics),
        "horizon_metrics": _horizon_metric_records(evaluation.horizon_metrics),
    }


def evaluation_summary(
    evaluation: RollingOriginRidgeEvaluation,
    *,
    input_path: Path,
    row_count: int,
) -> dict[str, Any]:
    validation_windows = [
        _window_summary(window_evaluation) for window_evaluation in evaluation.validation_windows
    ]
    return {
        "input_path": str(input_path),
        "row_count": row_count,
        "validation_window_count": len(validation_windows),
        "validation_windows": validation_windows,
        "final_test": _window_summary(evaluation.final_test),
    }


def run_ridge_evaluation(
    supervised_csv: Path,
    *,
    validation_months: int,
    alpha: float,
) -> dict[str, Any]:
    supervised_frame = _read_supervised_csv(supervised_csv)
    splits = build_rolling_origin_splits(
        supervised_frame,
        validation_months=validation_months,
    )
    evaluation = evaluate_rolling_origin_ridge(
        supervised_frame,
        splits,
        model_config=RidgeModelConfig(alpha=alpha),
    )
    return evaluation_summary(
        evaluation, input_path=supervised_csv, row_count=len(supervised_frame)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validation_months = _positive_integer(
            args.validation_months,
            name="validation-months",
        )
        alpha = _positive_float(args.alpha, name="alpha")
        summary = run_ridge_evaluation(
            args.supervised_csv,
            validation_months=validation_months,
            alpha=alpha,
        )
    except (ModelTrainingError, RidgeEvaluationCliError, SplitConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, sort_keys=True))
    return 0
