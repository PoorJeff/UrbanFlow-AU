from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from urbanflow.modeling.supervised_dataset import SupervisedSnapshotWriteError
from urbanflow.modeling.supervised_dataset_cli import build_parser, main


def write_hourly_snapshot(tmp_path: Path) -> Path:
    timestamps = pd.date_range("2025-05-01 00:00", periods=200, freq="h")
    frame = pd.DataFrame(
        {
            "id": [f"101{timestamp:%Y%m%d%H}" for timestamp in timestamps],
            "location_id": [101] * len(timestamps),
            "sensing_date": timestamps.strftime("%Y-%m-%d"),
            "hourday": timestamps.hour,
            "direction_1": [4] * len(timestamps),
            "direction_2": [6] * len(timestamps),
            "pedestriancount": [10] * len(timestamps),
            "sensor_name": ["Demo sensor"] * len(timestamps),
            "location": ["-37.8, 144.9"] * len(timestamps),
        }
    )
    path = tmp_path / "records.csv"
    frame.to_csv(path, index=False)
    return path


def write_matching_manifest(snapshot: Path, path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "dataset": "hourly_counts",
        "source_url": "https://example.test/hourly-counts",
        "extracted_at": "20250401T000000Z",
        "record_count": len(pd.read_csv(snapshot)),
        "source_total_count": 999,
        "snapshot_path": str(snapshot),
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_calendar(tmp_path: Path, *, end: str = "2025-06-01") -> Path:
    path = tmp_path / "holidays.json"
    path.write_text(
        json.dumps(
            {
                "coverage_start": "2025-01-01",
                "coverage_end": end,
                "public_holidays": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def valid_arguments(tmp_path: Path) -> list[str]:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    return [
        str(snapshot),
        str(manifest),
        str(tmp_path / "supervised.csv"),
        "--holiday-calendar",
        str(write_calendar(tmp_path)),
    ]


def test_parser_requires_three_paths_and_a_holiday_calendar() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as missing_calendar:
        parser.parse_args(["records.csv", "source.json", "supervised.csv"])
    with pytest.raises(SystemExit) as extra_path:
        parser.parse_args(
            [
                "records.csv",
                "source.json",
                "supervised.csv",
                "unexpected.csv",
                "--holiday-calendar",
                "holidays.json",
            ]
        )

    args = parser.parse_args(
        [
            "records.csv",
            "source.json",
            "supervised.csv",
            "--holiday-calendar",
            "holidays.json",
        ]
    )

    assert missing_calendar.value.code == 2
    assert extra_path.value.code == 2
    assert args.snapshot_path == Path("records.csv")
    assert args.manifest_path == Path("source.json")
    assert args.output_csv == Path("supervised.csv")
    assert args.holiday_calendar == Path("holidays.json")
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert option_strings == {"-h", "--help", "--holiday-calendar"}


def test_main_writes_only_json_for_a_valid_local_build(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = valid_arguments(tmp_path)
    output = Path(arguments[2])

    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "output_path": str(output),
        "snapshot_sha256": hashlib.sha256(Path(arguments[0]).read_bytes()).hexdigest(),
        "source_row_count": 200,
        "supervised_row_count": 4800,
        "training_row_count": 4500,
        "validation_warning_count": 1,
    }
    assert output.exists()


@pytest.mark.parametrize(
    "input_error",
    [
        "missing_snapshot",
        "malformed_manifest",
        "invalid_calendar",
        "insufficient_calendar_coverage",
        "existing_output",
    ],
)
def test_main_returns_two_for_local_input_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    input_error: str,
) -> None:
    arguments = valid_arguments(tmp_path)
    snapshot = Path(arguments[0])
    manifest = Path(arguments[1])
    output = Path(arguments[2])
    calendar = Path(arguments[arguments.index("--holiday-calendar") + 1])

    if input_error == "missing_snapshot":
        snapshot.unlink()
    elif input_error == "malformed_manifest":
        manifest.write_text("{not-json", encoding="utf-8")
    elif input_error == "invalid_calendar":
        calendar.write_text("{not-json", encoding="utf-8")
    elif input_error == "insufficient_calendar_coverage":
        write_calendar(tmp_path, end="2025-05-09")
    else:
        output.write_text("existing output", encoding="utf-8")

    exit_code = main(arguments)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: ")


def test_main_returns_one_for_output_write_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urbanflow.modeling.supervised_dataset_cli as module

    def fail_write(*args: object, **kwargs: object) -> object:
        raise SupervisedSnapshotWriteError("disk is full")

    monkeypatch.setattr(module, "build_supervised_csv_from_hourly_snapshot", fail_write)

    exit_code = main(valid_arguments(tmp_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "error: disk is full\n"


def test_build_supervised_csv_script_help() -> None:
    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, repository_root / "scripts" / "build_supervised_csv.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--holiday-calendar" in result.stdout
