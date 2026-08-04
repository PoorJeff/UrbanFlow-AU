from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from urbanflow.modeling.lightgbm_artifact import HolidayCalendar
from urbanflow.modeling.supervised_csv import read_supervised_csv
from urbanflow.modeling.supervised_dataset import (
    SupervisedSnapshotBuildError,
    SupervisedSnapshotWriteError,
    build_supervised_csv_from_hourly_snapshot,
)


def write_hourly_snapshot(
    tmp_path: Path, *, periods: int = 200, start: str = "2025-05-01 00:00"
) -> Path:
    timestamps = pd.date_range(start, periods=periods, freq="h")
    frame = pd.DataFrame(
        {
            "id": [f"101{stamp:%Y%m%d%H}" for stamp in timestamps],
            "location_id": [101] * periods,
            "sensing_date": timestamps.strftime("%Y-%m-%d"),
            "hourday": timestamps.hour,
            "direction_1": [4] * periods,
            "direction_2": [6] * periods,
            "pedestriancount": [10] * periods,
            "sensor_name": ["Demo sensor"] * periods,
            "location": ["-37.8, 144.9"] * periods,
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


def write_calendar(tmp_path: Path, *, end: str = "2025-06-01") -> HolidayCalendar:
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
    return HolidayCalendar.from_json_file(path)


def build(snapshot: Path, manifest: Path, output: Path, calendar: HolidayCalendar) -> object:
    return build_supervised_csv_from_hourly_snapshot(
        snapshot, manifest, output, holiday_calendar=calendar
    )


def assert_no_output_or_temp(output: Path) -> None:
    assert not os.path.lexists(output)
    assert not list(output.parent.glob(f".{output.name}-*"))


def test_builder_writes_direct_rows_from_a_verified_snapshot(tmp_path: Path) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    output = tmp_path / "supervised.csv"

    result = build(snapshot, manifest, output, write_calendar(tmp_path))

    round_tripped = read_supervised_csv(output)
    assert result.source_row_count == 200
    assert result.supervised_row_count == 4800
    assert result.training_row_count == 4500
    assert result.validation_warning_count == 1
    assert result.snapshot_sha256 == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert set(round_tripped["forecast_horizon"]) == set(range(1, 25))
    assert str(round_tripped["forecast_origin_at"].dtype) == "datetime64[ns, UTC]"
    assert round_tripped["temperature"].isna().all()
    assert round_tripped["temperature_missing"].all()


def test_builder_retains_missing_panel_origin_across_dst_fallback(tmp_path: Path) -> None:
    snapshot = write_hourly_snapshot(tmp_path, start="2025-04-01 00:00")
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    output = tmp_path / "supervised.csv"

    result = build(snapshot, manifest, output, write_calendar(tmp_path))
    round_tripped = read_supervised_csv(output)
    fallback_origin_rows = round_tripped.loc[
        round_tripped["forecast_origin_at"] == pd.Timestamp("2025-04-05T16:00:00Z")
    ]

    assert result.supervised_row_count == 4824
    assert result.training_row_count == 4500
    assert len(fallback_origin_rows) == 24
    assert fallback_origin_rows["pedestrian_count_missing"].all()
    assert str(round_tripped["forecast_origin_at"].dtype) == "datetime64[ns, UTC]"


def test_builder_reads_and_parses_hourly_snapshot_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    output = tmp_path / "supervised.csv"
    import urbanflow.modeling.supervised_dataset as module

    original_read_bytes = Path.read_bytes
    calls = 0

    def spy_read_bytes(path: Path) -> bytes:
        nonlocal calls
        if path == snapshot:
            calls += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", spy_read_bytes)
    original_read_csv = module.pd.read_csv
    parsed_snapshot = 0

    def spy_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal parsed_snapshot
        if args and getattr(args[0], "getvalue", lambda: None)() == original_read_bytes(snapshot):
            parsed_snapshot += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(module.pd, "read_csv", spy_read_csv)

    build(snapshot, manifest, output, write_calendar(tmp_path))

    assert calls == 1
    assert parsed_snapshot == 1


def test_manifest_mismatch_after_snapshot_mutation_creates_no_output(tmp_path: Path) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    snapshot.write_bytes(snapshot.read_bytes() + b"\n")
    output = tmp_path / "supervised.csv"

    with pytest.raises(SupervisedSnapshotBuildError, match="manifest does not match"):
        build(snapshot, manifest, output, write_calendar(tmp_path))

    assert_no_output_or_temp(output)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("schema_version"),
        lambda payload: payload.update(schema_version=True),
        lambda payload: payload.update(schema_version=2),
        lambda payload: payload.update(dataset="other"),
        lambda payload: payload.update(source_url=""),
        lambda payload: payload.update(snapshot_path=""),
        lambda payload: payload.update(extracted_at="2025-04-01"),
        lambda payload: payload.update(record_count=True),
        lambda payload: payload.update(record_count=-1),
        lambda payload: payload.update(source_total_count=True),
        lambda payload: payload.update(source_total_count=-1),
        lambda payload: payload.update(snapshot_sha256="A" * 64),
        lambda payload: payload.update(snapshot_sha256="0" * 63),
        lambda payload: payload.update(record_count=201),
        lambda payload: payload.update(snapshot_sha256="0" * 64),
    ],
)
def test_builder_rejects_invalid_manifest_fields(tmp_path: Path, mutate: object) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutate(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "supervised.csv"

    with pytest.raises(
        SupervisedSnapshotBuildError, match="^hourly-count manifest does not match snapshot$"
    ):
        build(snapshot, manifest, output, write_calendar(tmp_path))

    assert_no_output_or_temp(output)


@pytest.mark.parametrize("contents", ["[]", "{", "not json"])
def test_builder_rejects_malformed_manifest(tmp_path: Path, contents: str) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = tmp_path / "source.json"
    manifest.write_text(contents, encoding="utf-8")
    output = tmp_path / "supervised.csv"

    with pytest.raises(
        SupervisedSnapshotBuildError, match="^hourly-count manifest does not match snapshot$"
    ):
        build(snapshot, manifest, output, write_calendar(tmp_path))

    assert_no_output_or_temp(output)


def test_builder_accepts_extra_manifest_fields_and_stale_stored_path(tmp_path: Path) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.update(snapshot_path="C:/deleted/old-worktree/records.csv", metadata={"ignored": True})
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "supervised.csv"

    result = build(snapshot, manifest, output, write_calendar(tmp_path))

    assert result.source_row_count == 200
    assert output.exists()


def test_builder_does_not_compare_source_total_count_to_record_count(tmp_path: Path) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_total_count"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    build(snapshot, manifest, tmp_path / "supervised.csv", write_calendar(tmp_path))


def test_builder_rejects_schema_and_direction_errors(tmp_path: Path) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    frame = pd.read_csv(snapshot)
    frame.loc[0, "direction_1"] = 5
    frame.loc[1, "hourday"] = 99
    frame.to_csv(snapshot, index=False)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    output = tmp_path / "supervised.csv"

    with pytest.raises(
        SupervisedSnapshotBuildError,
        match="hourly-count snapshot validation failed: SCHEMA_INVALID, DIRECTION_TOTAL_MISMATCH",
    ):
        build(snapshot, manifest, output, write_calendar(tmp_path))

    assert_no_output_or_temp(output)


def test_builder_rejects_header_only_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "records.csv"
    snapshot.write_text("id,location_id,sensing_date,hourday\n", encoding="utf-8")
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    output = tmp_path / "supervised.csv"

    with pytest.raises(SupervisedSnapshotBuildError, match="snapshot contains no rows"):
        build(snapshot, manifest, output, write_calendar(tmp_path))

    assert_no_output_or_temp(output)


def test_builder_rejects_empty_snapshot_before_output_creation(tmp_path: Path) -> None:
    snapshot = tmp_path / "records.csv"
    snapshot.write_bytes(b"")
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "hourly_counts",
                "source_url": "https://example.test/hourly-counts",
                "extracted_at": "20250401T000000Z",
                "record_count": 0,
                "source_total_count": 0,
                "snapshot_path": str(snapshot),
                "snapshot_sha256": hashlib.sha256(b"").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "supervised.csv"

    with pytest.raises(SupervisedSnapshotBuildError, match="could not read hourly-count snapshot"):
        build(snapshot, manifest, output, write_calendar(tmp_path))

    assert_no_output_or_temp(output)


def test_builder_rejects_duplicate_sensor_hour(tmp_path: Path) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    frame = pd.read_csv(snapshot)
    frame.loc[1, ["sensing_date", "hourday"]] = frame.loc[0, ["sensing_date", "hourday"]]
    frame.to_csv(snapshot, index=False)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")

    with pytest.raises(SupervisedSnapshotBuildError, match="duplicate sensor-hour rows"):
        build(snapshot, manifest, tmp_path / "supervised.csv", write_calendar(tmp_path))


def test_builder_preserves_other_source_warnings_and_missing_target_markers(tmp_path: Path) -> None:
    snapshot = write_hourly_snapshot(tmp_path, periods=48)
    frame = pd.read_csv(snapshot).drop(index=5).reset_index(drop=True)
    frame.loc[1, "id"] = frame.loc[0, "id"]
    frame.to_csv(snapshot, index=False)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    output = tmp_path / "supervised.csv"

    result = build(snapshot, manifest, output, write_calendar(tmp_path))
    supervised = read_supervised_csv(output)

    assert result.validation_warning_count == 2
    assert supervised.loc[
        supervised["target_observed_at"] == pd.Timestamp("2025-04-30T19:00:00Z"),
        "target_missing",
    ].all()


def test_builder_requires_calendar_through_final_target_date(tmp_path: Path) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    output = tmp_path / "supervised.csv"

    with pytest.raises(SupervisedSnapshotBuildError, match="holiday calendar does not cover"):
        build(snapshot, manifest, output, write_calendar(tmp_path, end="2025-04-09"))

    assert_no_output_or_temp(output)


def test_existing_destination_is_not_overwritten(tmp_path: Path) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    output = tmp_path / "supervised.csv"
    output.write_bytes(b"existing bytes")

    with pytest.raises(SupervisedSnapshotBuildError, match="output already exists"):
        build(snapshot, manifest, output, write_calendar(tmp_path))

    assert output.read_bytes() == b"existing bytes"


@pytest.mark.parametrize("failure", ["to_csv", "round_trip", "link"])
def test_write_failure_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    output = tmp_path / "supervised.csv"
    import urbanflow.modeling.supervised_dataset as module

    if failure == "to_csv":
        monkeypatch.setattr(
            pd.DataFrame, "to_csv", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no"))
        )
    elif failure == "round_trip":
        monkeypatch.setattr(
            module,
            "read_supervised_csv",
            lambda path: (_ for _ in ()).throw(module.SupervisedCsvError("bad")),
        )
    else:
        monkeypatch.setattr(
            module.os, "link", lambda source, destination: (_ for _ in ()).throw(OSError("no"))
        )

    with pytest.raises(SupervisedSnapshotWriteError, match="could not write supervised CSV"):
        build(snapshot, manifest, output, write_calendar(tmp_path))

    assert_no_output_or_temp(output)


def test_competing_publish_destination_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = write_hourly_snapshot(tmp_path)
    manifest = write_matching_manifest(snapshot, tmp_path / "source.json")
    output = tmp_path / "supervised.csv"
    import urbanflow.modeling.supervised_dataset as module

    def competing_link(source: Path, destination: Path) -> None:
        destination.write_bytes(b"competitor")
        raise FileExistsError

    monkeypatch.setattr(module.os, "link", competing_link)

    with pytest.raises(SupervisedSnapshotBuildError, match="output already exists"):
        build(snapshot, manifest, output, write_calendar(tmp_path))

    assert output.read_bytes() == b"competitor"
    assert not list(output.parent.glob(f".{output.name}-*"))
