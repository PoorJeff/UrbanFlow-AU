from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from urbanflow.modeling.lightgbm_cli import main


def supervised_rows() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-12-01 00:00",
        "2025-02-28 23:00",
        freq="h",
        tz="Australia/Melbourne",
    )
    values = [
        100.0 + float(index % 24) + float((index // 24) % 7) + float(index // (24 * 7))
        for index in range(len(timestamps))
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


def test_lightgbm_evaluation_cli_returns_json_summary(tmp_path, capsys) -> None:
    path = write_supervised_csv(tmp_path)

    exit_code = main(
        [
            str(path),
            "--validation-months",
            "1",
            "--n-estimators",
            "5",
            "--min-child-samples",
            "1",
        ]
    )

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
    final_test = payload["final_test"]
    assert final_test["seasonal_naive_overall"]["row_count"] == 672
    assert final_test["seasonal_naive_horizon_metrics"][0]["row_count"] == 672
    assert_finite_metric(final_test["seasonal_naive_overall"]["mae"])
    assert_finite_metric(final_test["seasonal_naive_overall"]["rmse"])
    assert_finite_metric(final_test["seasonal_naive_overall"]["wape"])
    assert final_test["model_comparison"]["lightgbm_wape"] == final_test["overall"]["wape"]
    assert (
        final_test["model_comparison"]["seasonal_naive_wape"]
        == final_test["seasonal_naive_overall"]["wape"]
    )
    assert_finite_metric(final_test["model_comparison"]["relative_wape_improvement"])


def test_lightgbm_evaluation_cli_returns_two_for_missing_input(tmp_path, capsys) -> None:
    exit_code = main([str(tmp_path / "missing.csv")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "CSV file does not exist" in captured.err


def test_lightgbm_evaluation_cli_preserves_reader_error_behavior(tmp_path, capsys) -> None:
    frame = supervised_rows()
    frame["forecast_origin_at"] = frame["forecast_origin_at"].astype(str)
    frame.loc[0, "forecast_origin_at"] = "2025-01-01T00:00:00"
    path = tmp_path / "naive_timestamp.csv"
    frame.to_csv(path, index=False)

    exit_code = main([str(path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "could not parse timestamp column: forecast_origin_at" in captured.err


def test_lightgbm_evaluation_cli_returns_two_for_invalid_options(tmp_path, capsys) -> None:
    path = write_supervised_csv(tmp_path)

    exit_code = main([str(path), "--validation-months", "0", "--n-estimators", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "validation-months must be greater than zero" in captured.err


def test_lightgbm_evaluation_cli_returns_two_for_conflicting_seasonal_naive_panel(
    tmp_path,
    capsys,
) -> None:
    frame = supervised_rows()
    duplicate = frame.iloc[[0]].copy()
    duplicate["target"] = duplicate["target"] + 1.0
    frame = pd.concat([frame, duplicate], ignore_index=True)
    path = tmp_path / "conflicting_supervised_rows.csv"
    frame.to_csv(path, index=False)

    exit_code = main([str(path), "--validation-months", "1", "--n-estimators", "5"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert (
        "conflicting target values for duplicate location_id and target_observed_at" in captured.err
    )


def test_lightgbm_evaluation_cli_returns_two_when_seasonal_naive_is_unavailable(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    path = write_supervised_csv(tmp_path)
    window = SimpleNamespace(
        name="validation_2025-01",
        start=pd.Timestamp("2025-01-01", tz="Australia/Melbourne"),
        end=pd.Timestamp("2025-02-01", tz="Australia/Melbourne"),
        train_end=pd.Timestamp("2025-01-01", tz="Australia/Melbourne"),
    )
    lightgbm_metrics = SimpleNamespace(row_count=1, mae=1.0, rmse=1.0, wape=0.1)
    seasonal_naive_metrics = SimpleNamespace(row_count=0, mae=None, rmse=None, wape=None)
    fake_window = SimpleNamespace(
        window=window,
        predictions=pd.DataFrame(),
        overall_metrics=lightgbm_metrics,
        horizon_metrics=pd.DataFrame(
            [{"forecast_horizon": 1, "row_count": 1, "mae": 1.0, "rmse": 1.0, "wape": 0.1}]
        ),
        model=SimpleNamespace(training_row_count=1),
        seasonal_naive_overall_metrics=seasonal_naive_metrics,
        seasonal_naive_horizon_metrics=pd.DataFrame(
            [{"forecast_horizon": 1, "row_count": 0, "mae": None, "rmse": None, "wape": None}]
        ),
        model_comparison=SimpleNamespace(
            lightgbm_wape=0.1,
            seasonal_naive_wape=None,
            relative_wape_improvement=None,
        ),
    )
    fake_evaluation = SimpleNamespace(validation_windows=(fake_window,), final_test=fake_window)

    monkeypatch.setattr(
        "urbanflow.modeling.lightgbm_cli.evaluate_rolling_origin_lightgbm",
        lambda *args, **kwargs: fake_evaluation,
    )

    exit_code = main([str(path), "--validation-months", "1"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Seasonal Naive baseline unavailable for all evaluation windows" in captured.err


def test_evaluate_lightgbm_baseline_script_help() -> None:
    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [
            sys.executable,
            repository_root / "scripts" / "evaluate_lightgbm_baseline.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Evaluate a local LightGBM baseline" in result.stdout
