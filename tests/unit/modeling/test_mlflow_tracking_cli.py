from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from urbanflow.modeling import mlflow_tracking_cli
from urbanflow.modeling.mlflow_tracking import MLflowRunResult, MLflowTrackingError


def test_mlflow_tracking_cli_returns_json_result(tmp_path, capsys, monkeypatch) -> None:
    summary_path = tmp_path / "lightgbm_evaluation.json"
    summary_path.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "lightgbm_evaluation.md"
    report_path.write_text("# LightGBM evaluation\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_track_evaluation_summary(
        model_name: str,
        summary_json_path: Path,
        *,
        report_path: Path | None,
        config,
    ) -> MLflowRunResult:
        calls.append(
            {
                "model_name": model_name,
                "summary_json_path": summary_json_path,
                "report_path": report_path,
                "experiment_name": config.experiment_name,
                "tracking_uri": config.tracking_uri,
                "extra_tags": config.extra_tags,
            }
        )
        return MLflowRunResult(
            run_id="run-123",
            experiment_id="experiment-456",
            tracking_uri="file:///tmp/mlruns",
        )

    monkeypatch.setattr(
        mlflow_tracking_cli,
        "track_evaluation_summary",
        fake_track_evaluation_summary,
    )

    exit_code = mlflow_tracking_cli.main(
        [
            "lightgbm",
            str(summary_path),
            "--report",
            str(report_path),
            "--experiment-name",
            "custom-experiment",
            "--tracking-uri",
            "file:///tmp/mlruns",
            "--tag",
            "owner=urbanflow",
            "--tag",
            "slice=local",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "experiment_id": "experiment-456",
        "run_id": "run-123",
        "tracking_uri": "file:///tmp/mlruns",
    }
    assert calls == [
        {
            "model_name": "lightgbm",
            "summary_json_path": summary_path,
            "report_path": report_path,
            "experiment_name": "custom-experiment",
            "tracking_uri": "file:///tmp/mlruns",
            "extra_tags": {"owner": "urbanflow", "slice": "local"},
        }
    ]


def test_mlflow_tracking_cli_rejects_invalid_tag(tmp_path, capsys) -> None:
    summary_path = tmp_path / "ridge_evaluation.json"
    summary_path.write_text("{}", encoding="utf-8")

    exit_code = mlflow_tracking_cli.main(
        ["ridge", str(summary_path), "--tag", "not-a-key-value-pair"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "tags must use key=value format" in captured.err


def test_mlflow_tracking_cli_returns_two_for_tracking_errors(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    summary_path = tmp_path / "ridge_evaluation.json"
    summary_path.write_text("{}", encoding="utf-8")

    def fake_track_evaluation_summary(*args, **kwargs) -> MLflowRunResult:
        raise MLflowTrackingError("summary JSON must contain an object")

    monkeypatch.setattr(
        mlflow_tracking_cli,
        "track_evaluation_summary",
        fake_track_evaluation_summary,
    )

    exit_code = mlflow_tracking_cli.main(["ridge", str(summary_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "summary JSON must contain an object" in captured.err


def test_track_modeling_evaluation_script_help() -> None:
    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [
            sys.executable,
            repository_root / "scripts" / "track_modeling_evaluation.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Log an existing Ridge or LightGBM evaluation artifact to MLflow" in (result.stdout)
    assert "does not run training" in result.stdout
