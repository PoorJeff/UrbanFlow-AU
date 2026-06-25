"""Data ingestion boundaries for UrbanFlow AU."""

from urbanflow.ingestion.sensor_location_pipeline import (
    SensorLocationIngestionResult,
    ingest_sensor_locations,
)

__all__ = ["SensorLocationIngestionResult", "ingest_sensor_locations"]
