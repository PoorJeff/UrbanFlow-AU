# Sensor Location Pipeline Design

## Purpose

Add a small Python function entry point that runs the existing sensor-location ingestion pieces as one reusable workflow:

```text
fetch records -> normalize records -> write JSON snapshot -> write manifest -> return result metadata
```

This slice makes the ingestion code useful to future CLI, Prefect, and scheduled jobs without adding those layers yet.

## Selected Approach

The selected approach is a plain Python function in `src/urbanflow/ingestion/sensor_location_pipeline.py`.

This was chosen over adding a CLI or Prefect flow now because:

- the project already has tested low-level ingestion pieces;
- a function boundary is easiest to test without network access;
- a future CLI, notebook, Prefect flow, or API route can call the same function;
- it avoids adding user-interface and orchestration decisions before the data contract is settled.

## Public API

The module will expose:

- `SENSOR_LOCATIONS_SOURCE_DATASET`: the City of Melbourne dataset slug.
- `SENSOR_LOCATIONS_SNAPSHOT_DATASET`: the short local dataset name used in snapshot and manifest paths.
- `SensorLocationIngestionResult`: an immutable dataclass containing:
  - `source_dataset`
  - `snapshot_dataset`
  - `source_url`
  - `extracted_at`
  - `source_total_count`
  - `record_count`
  - `snapshot_path`
  - `manifest_path`
- `ingest_sensor_locations(...)`: the orchestration function.

The function signature will be:

```python
def ingest_sensor_locations(
    *,
    api_client: SupportsDatasetRecords,
    raw_root_dir: Path,
    manifest_root_dir: Path,
    extracted_at: datetime | None = None,
    page_limit: int = 100,
) -> SensorLocationIngestionResult:
    ...
```

`api_client` is injected so tests can use a fake client and production code can use `MelbourneApiClient`.

## Data Flow

1. Fetch records from `pedestrian-counting-system-sensor-locations`.
2. Normalize source records with `normalize_sensor_locations`.
3. Choose `extracted_at`:
   - use the caller-provided timestamp when supplied;
   - otherwise use the current UTC time.
4. Write the normalized records to:

```text
<raw_root_dir>/melbourne/sensor_locations/extracted_at=<YYYYMMDDTHHMMSSZ>/records.json
```

5. Write the manifest to:

```text
<manifest_root_dir>/sensor_locations/<YYYYMMDDTHHMMSSZ>.json
```

6. Return paths and counts in `SensorLocationIngestionResult`.

## Error Handling

The pipeline does not swallow lower-level errors.

- API errors from `MelbourneApiClient` propagate.
- Sensor parsing errors from `normalize_sensor_locations` propagate.
- Snapshot and manifest `FileExistsError` propagates to preserve immutability.

If normalization fails, the function has not written snapshot or manifest files. This keeps partial outputs out of local data directories.

## Testing

Tests will use a fake API client returning `DatasetRecords`; no live network calls are allowed.

Required tests:

- The pipeline passes the correct source dataset slug and page limit to the client.
- The pipeline writes a normalized snapshot and manifest with matching record counts and source total count.
- The returned result references the generated snapshot and manifest paths.
- Invalid source records fail before any output file is written.

## Out of Scope

- Command-line interface.
- Prefect flow.
- Live API smoke command.
- Hourly-count ingestion.
- Parquet, pandas, database writes, and data-quality reports.

## Success Criteria

- The function entry point can run the existing ingestion pieces together in tests.
- The implementation remains network-independent in CI.
- The result object gives later orchestration layers enough metadata to report what happened.
- Existing tests continue to pass.
