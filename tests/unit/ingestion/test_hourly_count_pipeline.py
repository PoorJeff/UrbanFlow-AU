import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from urbanflow.ingestion.hourly_count_pipeline import ingest_hourly_counts
from urbanflow.ingestion.hourly_counts import (
    HOURLY_COUNT_COLUMNS,
    HOURLY_COUNTS_SOURCE_DATASET,
    HourlyCountDateRange,
    HourlyCountIngestionError,
)
from urbanflow.ingestion.melbourne_api import DatasetRecordCount

EXTRACTED_AT = datetime(2026, 6, 27, 8, 0, 5, tzinfo=UTC)
DATE_RANGE = HourlyCountDateRange(
    start_date=date(2025, 1, 1),
    end_date=date(2025, 1, 1),
)
CSV_BYTES = (
    b"id,location_id,sensing_date,hourday,direction_1,direction_2,"
    b"pedestriancount,sensor_name,location\n"
    b'51120250101,51,2025-01-01,1,100,79,179,Fra118_T,"-37.8, 144.9"\n'
)


class FakeHourlyApiClient:
    def __init__(self, *, total_count: int, csv_bytes: bytes = CSV_BYTES) -> None:
        self.total_count = total_count
        self.csv_bytes = csv_bytes
        self.count_calls: list[tuple[str, str | None]] = []
        self.export_calls: list[tuple[str, tuple[str, ...], str | None, Path]] = []

    def count_records(self, dataset: str, *, where: str | None = None) -> DatasetRecordCount:
        self.count_calls.append((dataset, where))
        return DatasetRecordCount(
            dataset=dataset,
            source_url=f"https://example.test/{dataset}/records",
            total_count=self.total_count,
        )

    def export_url(self, dataset: str, *, export_format: str) -> str:
        return f"https://example.test/{dataset}/exports/{export_format}"

    def export_csv(
        self,
        dataset: str,
        *,
        output_path: Path,
        select: tuple[str, ...],
        where: str | None = None,
    ) -> None:
        self.export_calls.append((dataset, select, where, output_path))
        output_path.write_bytes(self.csv_bytes)


def test_ingest_hourly_counts_writes_snapshot_manifest_and_returns_metadata(
    tmp_path: Path,
) -> None:
    fake_client = FakeHourlyApiClient(total_count=1)

    result = ingest_hourly_counts(
        api_client=fake_client,
        raw_root_dir=tmp_path / "raw",
        manifest_root_dir=tmp_path / "manifests",
        date_range=DATE_RANGE,
        extracted_at=EXTRACTED_AT,
    )

    expected_where = "sensing_date >= date'2025-01-01' AND sensing_date <= date'2025-01-01'"
    assert fake_client.count_calls == [(HOURLY_COUNTS_SOURCE_DATASET, expected_where)]
    assert fake_client.export_calls[0][:3] == (
        HOURLY_COUNTS_SOURCE_DATASET,
        HOURLY_COUNT_COLUMNS,
        expected_where,
    )
    assert result.source_total_count == 1
    assert result.record_count == 1
    assert result.snapshot_path.read_bytes() == CSV_BYTES
    assert result.snapshot_path.as_posix().endswith(
        "melbourne/hourly_counts/extracted_at=20260627T080005Z/records.csv"
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "hourly_counts"
    assert manifest["record_count"] == 1
    assert manifest["source_total_count"] == 1
    assert manifest["metadata"] == {
        "date_range": {"end": "2025-01-01", "start": "2025-01-01"},
        "selected_columns": list(HOURLY_COUNT_COLUMNS),
        "sensor_filter": "all",
        "snapshot_format": "csv",
        "source_dataset": HOURLY_COUNTS_SOURCE_DATASET,
        "source_where": expected_where,
    }


def test_ingest_hourly_counts_rejects_empty_source_range(tmp_path: Path) -> None:
    fake_client = FakeHourlyApiClient(total_count=0)

    with pytest.raises(HourlyCountIngestionError, match="No hourly-count rows"):
        ingest_hourly_counts(
            api_client=fake_client,
            raw_root_dir=tmp_path / "raw",
            manifest_root_dir=tmp_path / "manifests",
            date_range=DATE_RANGE,
            extracted_at=EXTRACTED_AT,
        )

    assert fake_client.export_calls == []
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "manifests").exists()


def test_ingest_hourly_counts_rejects_count_mismatch_before_manifest(
    tmp_path: Path,
) -> None:
    fake_client = FakeHourlyApiClient(total_count=2)

    with pytest.raises(HourlyCountIngestionError, match="row count"):
        ingest_hourly_counts(
            api_client=fake_client,
            raw_root_dir=tmp_path / "raw",
            manifest_root_dir=tmp_path / "manifests",
            date_range=DATE_RANGE,
            extracted_at=EXTRACTED_AT,
        )

    assert list((tmp_path / "raw").rglob("records.csv")) == []
    assert not (tmp_path / "manifests").exists()
