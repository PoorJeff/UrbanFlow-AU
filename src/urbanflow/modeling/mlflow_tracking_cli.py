from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from urbanflow.modeling.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    MLflowTrackingConfig,
    MLflowTrackingError,
    track_evaluation_summary,
)


class MLflowTrackingCliError(ValueError):
    """Raised when MLflow tracking CLI input is invalid."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Log an existing Ridge or LightGBM evaluation artifact to MLflow; "
            "this command does not run training."
        )
    )
    parser.add_argument(
        "model_name",
        choices=("ridge", "lightgbm"),
        help="Model family that produced the evaluation summary.",
    )
    parser.add_argument(
        "summary_json",
        type=Path,
        help="Existing evaluation JSON summary to log.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional Markdown evaluation report to log as an artifact.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name to set or create.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI. When omitted, MLflow uses its active default.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra MLflow tag. May be supplied multiple times.",
    )
    return parser


def _parse_extra_tags(tag_values: Sequence[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for tag_value in tag_values:
        if "=" not in tag_value:
            raise MLflowTrackingCliError("tags must use key=value format")
        key, value = tag_value.split("=", 1)
        if not key:
            raise MLflowTrackingCliError("tag keys must not be empty")
        tags[key] = value
    return tags


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = track_evaluation_summary(
            args.model_name,
            args.summary_json,
            report_path=args.report,
            config=MLflowTrackingConfig(
                experiment_name=args.experiment_name,
                tracking_uri=args.tracking_uri,
                extra_tags=_parse_extra_tags(args.tag),
            ),
        )
    except (MLflowTrackingCliError, MLflowTrackingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "experiment_id": result.experiment_id,
                "run_id": result.run_id,
                "tracking_uri": result.tracking_uri,
            },
            sort_keys=True,
        )
    )
    return 0
