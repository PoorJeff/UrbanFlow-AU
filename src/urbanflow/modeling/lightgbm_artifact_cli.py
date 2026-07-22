from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from urbanflow.modeling.feature_matrix import ModelTrainingError
from urbanflow.modeling.lightgbm import LightGBMModelConfig
from urbanflow.modeling.lightgbm_artifact import (
    HolidayCalendar,
    LightGBMArtifactError,
    LightGBMArtifactManifest,
    LightGBMArtifactSerializationError,
    export_lightgbm_artifact,
)
from urbanflow.modeling.supervised_csv import (
    SupervisedCsvError,
    read_supervised_csv_snapshot,
)


class LightGBMArtifactCliError(ValueError):
    """Raised when local artifact export CLI input is invalid."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a local final-fit LightGBM artifact from supervised feature rows."
    )
    parser.add_argument(
        "supervised_csv",
        type=Path,
        help="CSV containing already-built supervised feature rows.",
    )
    parser.add_argument(
        "output_directory",
        help="New local directory that will contain the artifact bundle.",
    )
    parser.add_argument(
        "--holiday-calendar",
        type=Path,
        required=True,
        help="Local JSON holiday calendar with explicit date coverage.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Positive number of LightGBM boosting rounds.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Positive LightGBM learning rate.",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="Positive maximum number of leaves in one LightGBM tree.",
    )
    parser.add_argument(
        "--min-child-samples",
        type=int,
        default=20,
        help="Positive minimum number of data points in a LightGBM leaf.",
    )
    parser.add_argument(
        "--evaluation-summary-path",
        default=None,
        help="Optional reference to the evaluation summary used to select this model.",
    )
    return parser


def _positive_integer(value: int, *, name: str) -> int:
    if value <= 0:
        raise LightGBMArtifactCliError(f"{name} must be greater than zero")
    return value


def _positive_float(value: float, *, name: str) -> float:
    if value <= 0:
        raise LightGBMArtifactCliError(f"{name} must be greater than zero")
    return value


def run_artifact_export(
    supervised_csv: Path,
    output_directory: str | Path,
    *,
    holiday_calendar_path: Path,
    model_config: LightGBMModelConfig,
    evaluation_summary_path: str | None,
) -> LightGBMArtifactManifest:
    supervised_frame, source_csv_sha256 = read_supervised_csv_snapshot(supervised_csv)
    holiday_calendar = HolidayCalendar.from_json_file(holiday_calendar_path)
    return export_lightgbm_artifact(
        supervised_frame,
        source_csv_sha256=source_csv_sha256,
        output_directory=output_directory,
        holiday_calendar=holiday_calendar,
        model_config=model_config,
        evaluation_summary_path=evaluation_summary_path,
    )


def _manifest_summary(
    manifest: LightGBMArtifactManifest,
    *,
    output_directory: str | Path,
) -> dict[str, object]:
    return {
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "training_row_count": manifest.training_row_count,
        "trained_through_at": manifest.trained_through_at.isoformat(),
        "output_directory": str(output_directory),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        model_config = LightGBMModelConfig(
            n_estimators=_positive_integer(args.n_estimators, name="n-estimators"),
            learning_rate=_positive_float(args.learning_rate, name="learning-rate"),
            num_leaves=_positive_integer(args.num_leaves, name="num-leaves"),
            min_child_samples=_positive_integer(
                args.min_child_samples,
                name="min-child-samples",
            ),
        )
        manifest = run_artifact_export(
            args.supervised_csv,
            args.output_directory,
            holiday_calendar_path=args.holiday_calendar,
            model_config=model_config,
            evaluation_summary_path=args.evaluation_summary_path,
        )
    except (
        LightGBMArtifactCliError,
        SupervisedCsvError,
        LightGBMArtifactError,
        ModelTrainingError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except LightGBMArtifactSerializationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            _manifest_summary(manifest, output_directory=args.output_directory), sort_keys=True
        )
    )
    return 0
