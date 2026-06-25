from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from urbanflow.ingestion.manifests import write_manifest
from urbanflow.ingestion.melbourne_api import DatasetRecords
from urbanflow.ingestion.sensor_locations import normalize_sensor_locations
from urbanflow.ingestion.snapshots import write_json_snapshot

SENSOR_LOCATIONS_SOURCE_DATASET = "pedestrian-counting-system-sensor-locations"
SENSOR_LOCATIONS_SNAPSHOT_DATASET = "sensor_locations"


class SupportsDatasetRecords(Protocol):
    def fetch_all_records(self, dataset: str, *, limit: int = 100) -> DatasetRecords: ...


@dataclass(frozen=True)
class SensorLocationIngestionResult:
    source_dataset: str
    snapshot_dataset: str
    source_url: str
    extracted_at: datetime
    source_total_count: int
    record_count: int
    snapshot_path: Path
    manifest_path: Path


def ingest_sensor_locations(
    *,
    api_client: SupportsDatasetRecords,
    raw_root_dir: Path,
    manifest_root_dir: Path,
    extracted_at: datetime | None = None,
    page_limit: int = 100,
) -> SensorLocationIngestionResult:
    if page_limit <= 0:
        raise ValueError("page_limit must be greater than zero")

    extraction_time = extracted_at or datetime.now(UTC)
    dataset_records = api_client.fetch_all_records(
        SENSOR_LOCATIONS_SOURCE_DATASET,
        limit=page_limit,
    )
    normalized_records = normalize_sensor_locations(dataset_records.records)
    snapshot_path = write_json_snapshot(
        records=normalized_records,
        root_dir=raw_root_dir,
        dataset=SENSOR_LOCATIONS_SNAPSHOT_DATASET,
        extracted_at=extraction_time,
    )
    manifest_path = write_manifest(
        root_dir=manifest_root_dir,
        dataset=SENSOR_LOCATIONS_SNAPSHOT_DATASET,
        source_url=dataset_records.source_url,
        extracted_at=extraction_time,
        record_count=len(normalized_records),
        source_total_count=dataset_records.total_count,
        snapshot_path=snapshot_path,
    )

    return SensorLocationIngestionResult(
        source_dataset=SENSOR_LOCATIONS_SOURCE_DATASET,
        snapshot_dataset=SENSOR_LOCATIONS_SNAPSHOT_DATASET,
        source_url=dataset_records.source_url,
        extracted_at=extraction_time,
        source_total_count=dataset_records.total_count,
        record_count=len(normalized_records),
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
    )
