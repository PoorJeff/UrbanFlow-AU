# Sensor Locations Ingestion Design

## Purpose

Build the first functional data-ingestion slice for UrbanFlow AU: fetch City of Melbourne pedestrian sensor-location records through the official API, validate and normalize the records, write an immutable local JSON snapshot, and generate a machine-readable manifest.

This slice creates the reusable ingestion boundary for later hourly-count ingestion without introducing databases, orchestration, Pandera, pandas, Parquet, or modeling yet.

## Current Source Contract

The source endpoint is:

```text
https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/pedestrian-counting-system-sensor-locations/records
```

On 2026-06-25, a live check with `limit=2&offset=0` returned `total_count=136`. A representative record included these fields:

- `location_id`
- `sensor_description`
- `sensor_name`
- `installation_date`
- `note`
- `location_type`
- `status`
- `direction_1`
- `direction_2`
- `latitude`
- `longitude`
- `location`

The parser treats only these fields as required for this slice:

- `location_id`
- `sensor_description`
- `sensor_name`
- `installation_date`
- `status`
- `latitude`
- `longitude`

Optional source fields are preserved when present but do not block ingestion.

## Considered Approaches

### 1. Start with the large hourly-count dataset

This would prove the most important ingestion path immediately, but it requires date filters, large-export strategy, fixture design for high-volume pagination, and storage decisions. It is too broad for the first functional slice.

### 2. Build a generic ingestion framework first

This would create abstractions for all Melbourne datasets before handling real records. It risks speculative interfaces and slower feedback.

### 3. Implement sensor-location ingestion first (selected)

Sensor locations are small, stable, and required by the later database, API, map, and forecast features. This path validates the API client, pagination, normalization, snapshots, manifests, and tests with minimal data volume.

## Architecture

The implementation will add a focused ingestion package:

- `src/urbanflow/ingestion/melbourne_api.py`
  - Builds dataset-records URLs.
  - Fetches pages with `limit` and `offset`.
  - Stops when fetched records reach `total_count`.
  - Uses `httpx` for HTTP calls and bounded retry configuration through `tenacity`.
  - Raises a project-specific error for API, timeout, status-code, and malformed-response failures.

- `src/urbanflow/ingestion/sensor_locations.py`
  - Defines a `SensorLocation` dataclass.
  - Parses source dictionaries into normalized records.
  - Coerces `location_id` to `int` and coordinates to `float`.
  - Keeps `installation_date` as the source ISO date string for now.
  - Rejects missing required fields and invalid coordinates.

- `src/urbanflow/ingestion/snapshots.py`
  - Writes immutable JSON snapshots under a caller-provided root directory.
  - Uses deterministic JSON serialization so hashes are stable.
  - Does not overwrite an existing snapshot path.

- `src/urbanflow/ingestion/manifests.py`
  - Computes a SHA-256 hash of the snapshot bytes.
  - Writes a manifest with source URL, dataset name, extraction timestamp, record count, total count, snapshot path, and hash.

The package remains importable without network access. Tests use deterministic fixtures and temporary directories.

## Data Flow

```text
official records endpoint
  -> paginated HTTP client
  -> raw record dictionaries
  -> SensorLocation parser
  -> deterministic JSON snapshot
  -> manifest JSON with hash and extraction metadata
```

The live API is not called in CI. Network smoke checks can be added later as an explicit local command.

## Snapshot and Manifest Format

Snapshots will be JSON arrays of normalized sensor records. The default local destination is:

```text
data/raw/melbourne/sensor_locations/extracted_at=<YYYYMMDDTHHMMSSZ>/records.json
```

Raw snapshots stay untracked through the existing `data/raw/` ignore rule.

Manifests will be JSON objects. The default local destination is:

```text
data/manifests/sensor_locations/<YYYYMMDDTHHMMSSZ>.json
```

Manifest fields:

- `schema_version`
- `dataset`
- `source_url`
- `extracted_at`
- `record_count`
- `source_total_count`
- `snapshot_path`
- `snapshot_sha256`

Generated real manifests may be committed only when intentionally used as small reproducibility evidence.

## Error Handling

- HTTP timeout, connection failure, and non-success status codes raise `MelbourneApiError`.
- Missing `results` or `total_count` in an API response raises `MelbourneApiError`.
- Missing required sensor fields, non-numeric IDs, non-numeric coordinates, or out-of-range coordinates raise `SensorLocationParseError`.
- Existing snapshot or manifest paths are not overwritten; the writer raises `FileExistsError`.

## Testing

Tests will cover:

- Paginated API fetching across multiple fixed responses.
- Detection of malformed API responses.
- Normalization of a representative sensor-location record.
- Rejection of missing required fields.
- Rejection of invalid latitude and longitude.
- Snapshot JSON determinism.
- Manifest hash and record-count metadata.

The full quality gate remains:

```powershell
& .\.venv\Scripts\python.exe -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
```

## Out of Scope

- Hourly pedestrian-count ingestion.
- Historical large export strategy.
- pandas, Parquet, and Pandera.
- PostgreSQL, SQLAlchemy, Alembic, and idempotent database writes.
- Prefect orchestration.
- Dashboard, API service, modeling, MLflow, and monitoring.

## Success Criteria

- Sensor-location records can be fetched through deterministic paginated client tests.
- Source records can be normalized into typed `SensorLocation` objects.
- Invalid source records fail loudly instead of being silently accepted.
- Snapshot files are deterministic and immutable.
- Manifest files include enough metadata to trace the source, extraction time, record count, and snapshot hash.
- CI remains network-independent and fast.
