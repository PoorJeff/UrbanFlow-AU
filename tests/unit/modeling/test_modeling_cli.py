from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from urbanflow.modeling.cli import main


def supervised_rows() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-12-01 00:00",
        "2025-02-28 23:00",
        freq="h",
        tz="Australia/Melbourne",
    )
    values = [
        100.0 + float(index % 24) + float((index // 24) % 7) for index in range(len(timestamps))
    ]
    return pd.DataFrame(
        {
            "location_id": [101] * len(timestamps),
            "forecast_origin_at": timestamps - pd.Timedelta(hours=1),
            "forecast_horizon": [1] * len(timestamps),
            "target_observed_at": timestamps,
            "target": values,
            "target_missing": [False] * len(timestamps),
            "pedestrian_count": [value - 1.0 for value in values],
            "pedestrian_count_missing": [False] * len(timestamps),
            "lag_1": [value - 1.0 for value in values],
            "lag_24": [value - 2.0 for value in values],
            "lag_168": [value - 3.0 for value in values],
            "rolling_24_mean": [value - 1.5 for value in values],
            "rolling_24_std": [2.0] * len(timestamps),
            "rolling_168_mean": [value - 2.5 for value in values],
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
            "temperature": [20.0] * len(timestamps),
            "temperature_missing": [False] * len(timestamps),
            "rainfall": [0.0] * len(timestamps),
            "rainfall_missing": [False] * len(timestamps),
            "wind_speed": [12.0] * len(timestamps),
            "wind_speed_missing": [False] * len(timestamps),
        }
    )


def write_supervised_csv(tmp_path: Path) -> Path:
    path = tmp_path / "supervised_rows.csv"
    supervised_rows().to_csv(path, index=False)
    return path


def assert_finite_metric(value: object) -> None:
    assert isinstance(value, float)
    assert math.isfinite(value)


def test_ridge_evaluation_cli_returns_json_summary(tmp_path, capsys) -> None:
    path = write_supervised_csv(tmp_path)

    exit_code = main([str(path), "--validation-months", "1", "--alpha", "0.5"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["input_path"] == str(path)
    assert payload["row_count"] == len(supervised_rows())
    assert payload["validation_window_count"] == 1
    assert payload["validation_windows"][0]["name"] == "validation_2025-01"
    assert payload["validation_windows"][0]["training_row_count"] == 744
    assert payload["validation_windows"][0]["overall"]["row_count"] == 744
    assert payload["validation_windows"][0]["horizon_metrics"][0]["forecast_horizon"] == 1
    assert payload["final_test"]["name"] == "final_test_2025-02"
    assert payload["final_test"]["training_row_count"] == 1488
    assert payload["final_test"]["overall"]["row_count"] == 672
    assert payload["final_test"]["horizon_metrics"][0]["row_count"] == 672
    assert_finite_metric(payload["final_test"]["overall"]["mae"])
    assert_finite_metric(payload["final_test"]["overall"]["rmse"])
    assert_finite_metric(payload["final_test"]["overall"]["wape"])


def test_ridge_evaluation_cli_returns_two_for_missing_input(tmp_path, capsys) -> None:
    exit_code = main([str(tmp_path / "missing.csv")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "CSV file does not exist" in captured.err


def test_ridge_evaluation_cli_returns_two_for_invalid_options(tmp_path, capsys) -> None:
    path = write_supervised_csv(tmp_path)

    exit_code = main([str(path), "--validation-months", "0", "--alpha", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "validation-months must be greater than zero" in captured.err


def test_evaluate_ridge_baseline_script_help() -> None:
    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, repository_root / "scripts" / "evaluate_ridge_baseline.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Evaluate a local Ridge baseline" in result.stdout
