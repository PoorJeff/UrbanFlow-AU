from datetime import UTC, datetime

import pytest

from urbanflow.database.loaders import (
    DatabaseLoadError,
    load_hourly_counts_snapshot,
    load_sensor_locations_snapshot,
)
from urbanflow.validation.reports import ValidationIssue, ValidationReport


class FakeSession:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, statement) -> None:
        self.calls.append(statement)


def _passing_report(
    dataset: str,
    snapshot_path: str,
    warning_count: int = 0,
) -> ValidationReport:
    return ValidationReport(
        dataset=dataset,
        snapshot_path=snapshot_path,
        validated_at=datetime(2026, 6, 29, 12, tzinfo=UTC),
        row_count=1,
        warnings=tuple(
            ValidationIssue(code=f"WARN_{index}", message="warning")
            for index in range(warning_count)
        ),
    )


def _failing_report(dataset: str, snapshot_path: str) -> ValidationReport:
    return ValidationReport(
        dataset=dataset,
        snapshot_path=snapshot_path,
        validated_at=datetime(2026, 6, 29, 12, tzinfo=UTC),
        row_count=0,
        errors=(ValidationIssue(code="SCHEMA_INVALID", message="bad snapshot"),),
    )


def test_load_sensor_locations_snapshot_refuses_failed_validation(
    monkeypatch,
    tmp_path,
) -> None:
    snapshot_path = tmp_path / "records.json"
    monkeypatch.setattr(
        "urbanflow.database.loaders.validate_snapshot",
        lambda dataset, path: _failing_report(dataset, str(path)),
    )

    with pytest.raises(DatabaseLoadError, match="Validation failed"):
        load_sensor_locations_snapshot(FakeSession(), snapshot_path)


def test_load_sensor_locations_snapshot_calls_repository(monkeypatch, tmp_path) -> None:
    snapshot_path = tmp_path / "records.json"
    fake_session = FakeSession()
    captured = {}
    monkeypatch.setattr(
        "urbanflow.database.loaders.validate_snapshot",
        lambda dataset, path: _passing_report(dataset, str(path), warning_count=2),
    )
    monkeypatch.setattr(
        "urbanflow.database.loaders.sensor_rows_from_snapshot",
        lambda path: [{"location_id": 1}],
    )

    def fake_upsert_sensor_rows(session, rows):
        captured["session"] = session
        captured["rows"] = rows
        return 1

    monkeypatch.setattr(
        "urbanflow.database.loaders.upsert_sensor_rows",
        fake_upsert_sensor_rows,
    )

    result = load_sensor_locations_snapshot(fake_session, snapshot_path)

    assert result.dataset == "sensor_locations"
    assert result.row_count == 1
    assert result.validation_warning_count == 2
    assert captured == {"session": fake_session, "rows": [{"location_id": 1}]}


def test_load_hourly_counts_snapshot_calls_repository(monkeypatch, tmp_path) -> None:
    snapshot_path = tmp_path / "records.csv"
    fake_session = FakeSession()
    captured = {}
    monkeypatch.setattr(
        "urbanflow.database.loaders.validate_snapshot",
        lambda dataset, path: _passing_report(dataset, str(path)),
    )
    monkeypatch.setattr(
        "urbanflow.database.loaders.hourly_count_rows_from_snapshot",
        lambda path: [{"location_id": 1}],
    )

    def fake_upsert_hourly_rows(session, rows):
        captured["session"] = session
        captured["rows"] = rows
        return 1

    monkeypatch.setattr(
        "urbanflow.database.loaders.upsert_hourly_rows",
        fake_upsert_hourly_rows,
    )

    result = load_hourly_counts_snapshot(fake_session, snapshot_path)

    assert result.dataset == "hourly_counts"
    assert result.row_count == 1
    assert result.validation_warning_count == 0
    assert captured == {"session": fake_session, "rows": [{"location_id": 1}]}
