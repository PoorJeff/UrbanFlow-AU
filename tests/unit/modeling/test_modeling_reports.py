from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from urbanflow.modeling.report_cli import main as report_main
from urbanflow.modeling.reports import RidgeReportError, render_ridge_evaluation_report


def ridge_summary() -> dict[str, object]:
    return {
        "input_path": "data/modeling/supervised_rows.csv",
        "row_count": 1464,
        "validation_window_count": 1,
        "validation_windows": [
            {
                "name": "validation_2025-01",
                "start": "2025-01-01T00:00:00+11:00",
                "end": "2025-02-01T00:00:00+11:00",
                "train_end": "2025-01-01T00:00:00+11:00",
                "training_row_count": 744,
                "overall": {
                    "row_count": 744,
                    "mae": 1.23456,
                    "rmse": 1.75432,
                    "wape": 0.08123,
                },
                "horizon_metrics": [
                    {
                        "forecast_horizon": 1,
                        "row_count": 744,
                        "mae": 1.23456,
                        "rmse": 1.75432,
                        "wape": 0.08123,
                    }
                ],
            }
        ],
        "final_test": {
            "name": "final_test_2025-02",
            "start": "2025-02-01T00:00:00+11:00",
            "end": "2025-03-01T00:00:00+11:00",
            "train_end": "2025-02-01T00:00:00+11:00",
            "training_row_count": 1488,
            "overall": {
                "row_count": 672,
                "mae": 1.2,
                "rmse": 1.7,
                "wape": 0.07,
            },
            "horizon_metrics": [
                {
                    "forecast_horizon": 1,
                    "row_count": 672,
                    "mae": 1.2,
                    "rmse": 1.7,
                    "wape": 0.07,
                }
            ],
        },
    }


def test_render_ridge_evaluation_report_includes_core_sections() -> None:
    markdown = render_ridge_evaluation_report(ridge_summary())

    assert markdown.startswith("# Ridge Evaluation Report\n")
    assert "Source: `data/modeling/supervised_rows.csv`" in markdown
    assert "Rows evaluated: 1464" in markdown
    assert "Validation windows: 1" in markdown
    assert "## Final test" in markdown
    assert "Window: `final_test_2025-02`" in markdown
    assert "| Row count | 672 |" in markdown
    assert "| MAE | 1.2000 |" in markdown
    assert "## Validation windows" in markdown
    assert (
        "| validation_2025-01 | 2025-01-01T00:00:00+11:00 to "
        "2025-02-01T00:00:00+11:00 | 744 | 744 | 1.2346 | 1.7543 | 0.0812 |" in markdown
    )
    assert "## Final test by horizon" in markdown
    assert "| 1 | 672 | 1.2000 | 1.7000 | 0.0700 |" in markdown
    assert markdown.endswith("\n")


def test_render_ridge_evaluation_report_includes_mermaid_metric_charts() -> None:
    markdown = render_ridge_evaluation_report(ridge_summary())

    assert markdown.index("## Validation windows") < markdown.index(
        "## Metric comparison charts"
    )
    assert markdown.index("## Metric comparison charts") < markdown.index(
        "## Final test by horizon"
    )
    assert markdown.count("```mermaid\nxychart-beta") == 3

    expected_mae_chart = """```mermaid
xychart-beta
    title "MAE by evaluation window"
    x-axis ["validation_2025-01", "final_test_2025-02"]
    y-axis "MAE" 0 --> 1.3580
    bar [1.2346, 1.2000]
```"""
    expected_rmse_chart = """```mermaid
xychart-beta
    title "RMSE by evaluation window"
    x-axis ["validation_2025-01", "final_test_2025-02"]
    y-axis "RMSE" 0 --> 1.9298
    bar [1.7543, 1.7000]
```"""
    expected_wape_chart = """```mermaid
xychart-beta
    title "WAPE by evaluation window"
    x-axis ["validation_2025-01", "final_test_2025-02"]
    y-axis "WAPE" 0 --> 0.0894
    bar [0.0812, 0.0700]
```"""

    assert expected_mae_chart in markdown
    assert expected_rmse_chart in markdown
    assert expected_wape_chart in markdown


def test_render_ridge_evaluation_report_formats_missing_metrics_as_na() -> None:
    summary = deepcopy(ridge_summary())
    final_test = summary["final_test"]
    assert isinstance(final_test, dict)
    overall = final_test["overall"]
    assert isinstance(overall, dict)
    overall["wape"] = None

    markdown = render_ridge_evaluation_report(summary)

    assert "| WAPE | n/a |" in markdown


def test_render_ridge_evaluation_report_rejects_missing_required_field() -> None:
    summary = deepcopy(ridge_summary())
    final_test = summary["final_test"]
    assert isinstance(final_test, dict)
    overall = final_test["overall"]
    assert isinstance(overall, dict)
    del overall["mae"]

    with pytest.raises(
        RidgeReportError,
        match="missing required summary field: final_test.overall.mae",
    ):
        render_ridge_evaluation_report(summary)


def write_summary_json(tmp_path: Path) -> Path:
    path = tmp_path / "ridge_evaluation.json"
    path.write_text(json.dumps(ridge_summary()), encoding="utf-8")
    return path


def test_report_cli_writes_markdown_file(tmp_path, capsys) -> None:
    summary_path = write_summary_json(tmp_path)
    output_path = tmp_path / "reports" / "ridge_evaluation.md"

    exit_code = report_main([str(summary_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload == {"output_path": str(output_path)}
    assert "# Ridge Evaluation Report" in output_path.read_text(encoding="utf-8")


def test_report_cli_returns_two_when_output_exists_without_force(tmp_path, capsys) -> None:
    summary_path = write_summary_json(tmp_path)
    output_path = tmp_path / "ridge_evaluation.md"
    output_path.write_text("existing", encoding="utf-8")

    exit_code = report_main([str(summary_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "output file already exists" in captured.err
    assert output_path.read_text(encoding="utf-8") == "existing"


def test_report_cli_force_overwrites_existing_output(tmp_path, capsys) -> None:
    summary_path = write_summary_json(tmp_path)
    output_path = tmp_path / "ridge_evaluation.md"
    output_path.write_text("existing", encoding="utf-8")

    exit_code = report_main([str(summary_path), "--output", str(output_path), "--force"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "# Ridge Evaluation Report" in output_path.read_text(encoding="utf-8")


def test_render_ridge_evaluation_report_script_help() -> None:
    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [
            sys.executable,
            repository_root / "scripts" / "render_ridge_evaluation_report.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Render a Ridge evaluation Markdown report" in result.stdout


def test_checked_in_ridge_example_report_matches_renderer() -> None:
    repository_root = Path(__file__).parents[3]
    summary_path = (
        repository_root / "docs" / "examples" / "modeling" / "ridge_evaluation_summary.json"
    )
    report_path = repository_root / "docs" / "examples" / "modeling" / "ridge_evaluation_report.md"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert render_ridge_evaluation_report(summary) == report_path.read_text(encoding="utf-8")
