from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from urbanflow.ingestion.hourly_counts import (
    HOURLY_COUNT_COLUMNS,
    HOURLY_COUNTS_SNAPSHOT_DATASET,
    HOURLY_COUNTS_SOURCE_DATASET,
    HourlyCountDateRange,
    HourlyCountIngestionError,
    build_hourly_counts_where,
    count_csv_data_rows,
)
from urbanflow.ingestion.manifests import write_manifest
from urbanflow.ingestion.melbourne_api import DatasetRecordCount
from urbanflow.ingestion.snapshots import format_extracted_at, move_file_snapshot


class SupportsHourlyCountExport(Protocol):
    def count_records(self, dataset: str, *, where: str | None = None) -> DatasetRecordCount: ...

    def export_url(self, dataset: str, *, export_format: str) -> str: ...

    def export_csv(
        self,
        dataset: str,
        *,
        output_path: Path,
        select: tuple[str, ...],
        where: str | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class HourlyCountIngestionResult:
    source_dataset: str
    snapshot_dataset: str
    source_url: str
    extracted_at: datetime
    date_range: HourlyCountDateRange
    source_total_count: int
    record_count: int
    selected_columns: tuple[str, ...]
    snapshot_path: Path
    manifest_path: Path


def ingest_hourly_counts(
    *,
    api_client: SupportsHourlyCountExport,
    raw_root_dir: Path,
    manifest_root_dir: Path,
    date_range: HourlyCountDateRange,
    extracted_at: datetime | None = None,
) -> HourlyCountIngestionResult:
    extraction_time = extracted_at or datetime.now(UTC)
    source_where = build_hourly_counts_where(date_range)
    record_count_result = api_client.count_records(
        HOURLY_COUNTS_SOURCE_DATASET,
        where=source_where,
    )
    if record_count_result.total_count <= 0:
        raise HourlyCountIngestionError("No hourly-count rows found for the requested date range")

    timestamp = format_extracted_at(extraction_time)
    temp_dir = raw_root_dir / "melbourne" / HOURLY_COUNTS_SNAPSHOT_DATASET / "_tmp"
    temp_path = temp_dir / f"{timestamp}.records.csv.tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        api_client.export_csv(
            HOURLY_COUNTS_SOURCE_DATASET,
            output_path=temp_path,
            select=HOURLY_COUNT_COLUMNS,
            where=source_where,
        )
        snapshot_record_count = count_csv_data_rows(temp_path)
        if snapshot_record_count != record_count_result.total_count:
            raise HourlyCountIngestionError(
                "CSV export row count "
                f"{snapshot_record_count} did not match source count "
                f"{record_count_result.total_count}"
            )

        snapshot_path = move_file_snapshot(
            source_path=temp_path,
            root_dir=raw_root_dir,
            dataset=HOURLY_COUNTS_SNAPSHOT_DATASET,
            extracted_at=extraction_time,
            filename="records.csv",
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

    source_url = api_client.export_url(HOURLY_COUNTS_SOURCE_DATASET, export_format="csv")
    manifest_path = write_manifest(
        root_dir=manifest_root_dir,
        dataset=HOURLY_COUNTS_SNAPSHOT_DATASET,
        source_url=source_url,
        extracted_at=extraction_time,
        record_count=snapshot_record_count,
        source_total_count=record_count_result.total_count,
        snapshot_path=snapshot_path,
        metadata={
            "date_range": {
                "end": date_range.end_date.isoformat(),
                "start": date_range.start_date.isoformat(),
            },
            "selected_columns": list(HOURLY_COUNT_COLUMNS),
            "sensor_filter": "all",
            "snapshot_format": "csv",
            "source_dataset": HOURLY_COUNTS_SOURCE_DATASET,
            "source_where": source_where,
        },
    )

    return HourlyCountIngestionResult(
        source_dataset=HOURLY_COUNTS_SOURCE_DATASET,
        snapshot_dataset=HOURLY_COUNTS_SNAPSHOT_DATASET,
        source_url=source_url,
        extracted_at=extraction_time,
        date_range=date_range,
        source_total_count=record_count_result.total_count,
        record_count=snapshot_record_count,
        selected_columns=HOURLY_COUNT_COLUMNS,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
    )
