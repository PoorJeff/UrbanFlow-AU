import json
from pathlib import Path

from urbanflow.validation.cli import main


def test_validation_cli_returns_zero_for_passing_snapshot(tmp_path, capsys):
    snapshot_path = tmp_path / "records.json"
    snapshot_path.write_text(
        json.dumps(
            [
                {
                    "location_id": 1,
                    "sensor_description": "Bourke Street",
                    "sensor_name": "Sensor A",
                    "installation_date": None,
                    "status": "A",
                    "latitude": -37.81,
                    "longitude": 144.96,
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["sensor_locations", str(snapshot_path)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "sensor_locations"
    assert payload["passed"] is True
    assert payload["error_count"] == 0


def test_validation_cli_returns_one_for_validation_failure(tmp_path, capsys):
    snapshot_path = tmp_path / "records.csv"
    snapshot_path.write_text(
        "id,location_id,sensing_date,hourday,direction_1,direction_2,pedestriancount,sensor_name,location\n"
        'a,1,2025-01-01,24,2,3,9,Sensor A,"-37.81,144.96"\n',
        encoding="utf-8",
    )

    exit_code = main(["hourly_counts", str(snapshot_path)])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["error_count"] >= 1


def test_validation_cli_returns_two_for_read_failure(tmp_path, capsys):
    exit_code = main(["hourly_counts", str(tmp_path / "missing.csv")])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["error_count"] == 1


def test_validation_script_help():
    import subprocess
    import sys

    repository_root = Path(__file__).parents[3]
    result = subprocess.run(
        [sys.executable, repository_root / "scripts" / "validate_snapshot.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Validate a local UrbanFlow AU raw snapshot" in result.stdout
