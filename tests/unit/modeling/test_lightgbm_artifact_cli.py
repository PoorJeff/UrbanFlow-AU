from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from urbanflow.modeling import supervised_csv as supervised_csv_module
from urbanflow.modeling.lightgbm_artifact_cli import build_parser, main


def supervised_rows() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-04-05 00:00",
        periods=192,
        freq="h",
        tz="Australia/Melbourne",
    )
    index = pd.Series(range(len(timestamps)), dtype="float64")
    return pd.DataFrame(
        {
            "location_id": [101] * len(timestamps),
            "forecast_origin_at": timestamps,
            "forecast_horizon": [1] * len(timestamps),
            "target_observed_at": timestamps + pd.Timedelta(1, unit="h"),
            "target": 100.0 + index,
            "target_missing": [False] * len(timestamps),
            "pedestrian_count": 99.0 + index,
            "pedestrian_count_missing": [False] * len(timestamps),
            "lag_1": 98.0 + index,
            "lag_24": 97.0 + index,
            "lag_168": 96.0 + index,
            "rolling_24_mean": 95.0 + index,
            "rolling_24_std": [2.0] * len(timestamps),
            "rolling_168_mean": 94.0 + index,
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
            "temperature": [pd.NA] * len(timestamps),
            "temperature_missing": [True] * len(timestamps),
            "rainfall": [pd.NA] * len(timestamps),
            "rainfall_missing": [True] * len(timestamps),
            "wind_speed": [pd.NA] * len(timestamps),
            "wind_speed_missing": [True] * len(timestamps),
        }
    )


def write_supervised_csv(tmp_path: Path) -> Path:
    path = tmp_path / "supervised.csv"
    supervised_rows().to_csv(path, index=False)
    return path


def write_holiday_calendar(tmp_path: Path, *, malformed: bool = False) -> Path:
    path = tmp_path / "holidays.json"
    if malformed:
        path.write_text("{not-json", encoding="utf-8")
    else:
        path.write_text(
            json.dumps(
                {
                    "coverage_start": "2025-01-01",
                    "coverage_end": "2026-12-31",
                    "public_holidays": ["2025-01-27", "2026-01-26"],
                }
            ),
            encoding="utf-8",
        )
    return path


def valid_arguments(tmp_path: Path) -> list[str]:
    return [
        str(write_supervised_csv(tmp_path)),
        str(tmp_path / "artifact"),
        "--holiday-calendar",
        str(write_holiday_calendar(tmp_path)),
        "--n-estimators",
        "5",
        "--min-child-samples",
        "1",
    ]


def test_cli_exports_final_fit_artifact_and_prints_json_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "artifact"

    exit_code = main(valid_arguments(tmp_path))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["model_name"] == "lightgbm"
    assert payload["model_version"].startswith("lightgbm-")
    assert payload["training_row_count"] == 192
    assert payload["trained_through_at"].endswith("+00:00")
    assert payload["output_directory"] == str(output)
    assert set(payload) == {
        "model_name",
        "model_version",
        "output_directory",
        "trained_through_at",
        "training_row_count",
    }
    assert {path.name for path in output.iterdir()} == {"manifest.json", "model.joblib"}


def test_cli_returns_two_for_invalid_operator_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            str(tmp_path / "missing.csv"),
            str(tmp_path / "artifact"),
            "--holiday-calendar",
            str(tmp_path / "missing-holidays.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "error:" in captured.err


@pytest.mark.parametrize(
    "option",
    [
        ["--n-estimators", "0"],
        ["--learning-rate", "0"],
    ],
)
def test_cli_returns_two_for_non_positive_model_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: list[str],
) -> None:
    arguments = valid_arguments(tmp_path)
    arguments.extend(option)

    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "must be greater than zero" in captured.err
    assert not (tmp_path / "artifact").exists()


def test_cli_returns_two_for_malformed_holiday_calendar(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = valid_arguments(tmp_path)
    calendar = write_holiday_calendar(tmp_path, malformed=True)
    arguments[arguments.index("--holiday-calendar") + 1] = str(calendar)

    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "invalid holiday calendar" in captured.err
    assert not (tmp_path / "artifact").exists()


@pytest.mark.parametrize("column", ["target", "lag_1"])
def test_cli_returns_two_for_non_numeric_training_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    column: str,
) -> None:
    arguments = valid_arguments(tmp_path)
    source_path = Path(arguments[0])
    frame = pd.read_csv(source_path)
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = "not-a-number"
    frame.to_csv(source_path, index=False)

    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "error:" in captured.err
    assert not (tmp_path / "artifact").exists()


def test_cli_hashes_the_same_source_byte_snapshot_that_it_parses(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = valid_arguments(tmp_path)
    source_path = Path(arguments[0])
    original_bytes = source_path.read_bytes()
    changed_bytes = b"changed,on,disk\n1,2,3\n"
    original_read_csv = supervised_csv_module.pd.read_csv
    source_was_changed = False

    def read_csv_then_change_source(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal source_was_changed
        frame = original_read_csv(*args, **kwargs)
        if not source_was_changed:
            source_path.write_bytes(changed_bytes)
            source_was_changed = True
        return frame

    monkeypatch.setattr(supervised_csv_module.pd, "read_csv", read_csv_then_change_source)

    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    manifest = json.loads((tmp_path / "artifact" / "manifest.json").read_text(encoding="utf-8"))
    assert source_path.read_bytes() == changed_bytes
    assert manifest["training_data_sha256"] == hashlib.sha256(original_bytes).hexdigest()


def test_cli_returns_two_for_unreadable_supervised_csv_snapshot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = valid_arguments(tmp_path)
    source_path = Path(arguments[0])
    original_read_bytes = Path.read_bytes

    def reject_source_read(path: Path) -> bytes:
        if path == source_path:
            raise PermissionError("access denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_source_read)

    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "error:" in captured.err
    assert not (tmp_path / "artifact").exists()


def test_cli_returns_two_for_existing_destination(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = valid_arguments(tmp_path)
    (tmp_path / "artifact").mkdir()

    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "already exists" in captured.err


def test_cli_rejects_opaque_uri_output_without_creating_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = valid_arguments(tmp_path)
    arguments[1] = "s3:bucket/artifact"
    monkeypatch.chdir(tmp_path)

    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "local path" in captured.err
    assert not (tmp_path / "s3:bucket").exists()


def test_cli_returns_one_for_artifact_serialization_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dump(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(joblib, "dump", fail_dump)

    exit_code = main(valid_arguments(tmp_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error:" in captured.err
    assert not (tmp_path / "artifact").exists()


def test_output_directory_parser_preserves_raw_windows_drive_text() -> None:
    output = r"C:\models\artifact"

    args = build_parser().parse_args(
        ["supervised.csv", output, "--holiday-calendar", "holidays.json"]
    )

    assert args.output_directory == output
    assert isinstance(args.output_directory, str)


def test_export_lightgbm_artifact_script_help() -> None:
    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, repository_root / "scripts" / "export_lightgbm_artifact.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Export a local final-fit LightGBM artifact" in result.stdout
