# Hourly Count Ingestion Design

## Goal

Add the second Melbourne ingestion path: a bounded, repeatable hourly pedestrian-count
extraction that writes an immutable local CSV snapshot and a manifest with provenance
metadata.

This completes the ingestion side of the first planned delivery slice without adding
validation, database persistence, orchestration, feature engineering, or modeling yet.

## Source Context

The source dataset is City of Melbourne's `pedestrian-counting-system-monthly-counts-per-hour`
dataset:

https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/pedestrian-counting-system-monthly-counts-per-hour/records

The project requirements identify these required fields:

- `location_id`
- `sensing_date`
- `hourday`
- `pedestriancount`
- `sensor_name`
- `location`

The source currently also exposes `id`, `direction_1`, and `direction_2`. The hourly
snapshot will preserve `id`, `direction_1`, and `direction_2` because they are useful
for duplicate checks and later data-quality analysis, but downstream modeling will not
depend on them initially.

Live API checks on 2026-06-27 found two constraints that shape this design:

- The `records` endpoint is suitable for metadata and small previews, but not for
  million-row extraction. It returns at most 100 records per request and ordinary
  offset pagination is not a good fit for full-range export.
- The `exports/csv` endpoint accepts the same query filters and can return the bounded
  result set as CSV. This is the selected extraction path for hourly-count snapshots.

The requirements say the MVP should use the most recent three complete calendar years.
The implementation must not hard-code a total record count or assume the source already
contains every requested year. The ingestion command will require an explicit date range
or year, compute the source count dynamically, and fail clearly when the selected range
has no source rows.

## Considered Approaches

### 1. Reuse `fetch_all_records()` with offset pagination

This would reuse the existing API client shape, but it is the wrong tool for this data
volume. The source has over a million hourly rows, the API enforces small page sizes, and
large offset pagination would be slow and brittle.

### 2. Download CSV and immediately convert to pandas or Parquet

This would be closer to analytics work, but it mixes raw ingestion with validation and
processing. It also forces an early decision between pandas and polars before the
validation/database slice.

### 3. Download bounded CSV export as the raw immutable snapshot (selected)

This keeps the ingestion boundary small and operationally useful. The raw snapshot stays
close to the source, can be hashed and reproduced, and leaves schema validation,
deduplication, sensor selection, and Parquet/database decisions to the next slice.

## Architecture

The implementation will extend the existing ingestion package while keeping file
responsibilities narrow:

- `src/urbanflow/ingestion/melbourne_api.py`
  - Add a URL builder for dataset export endpoints.
  - Add a count query for a dataset and `where` clause using the `records` endpoint with
    `limit=0`.
  - Add a CSV export downloader that streams response bytes to a caller-provided file
    path without loading the full dataset into memory.
- `src/urbanflow/ingestion/hourly_counts.py`
  - Own hourly-count date-range validation, selected source columns, source `where`
    clause creation, and CSV row counting.
  - Raise a domain-specific error for invalid ranges, empty ranges, malformed CSV
    exports, or count mismatches.
- `src/urbanflow/ingestion/hourly_count_pipeline.py`
  - Orchestrate: validate requested range, compute source total count, download the CSV
    export, count snapshot rows, write a manifest, and return typed result metadata.
- `src/urbanflow/ingestion/hourly_count_cli.py`
  - Provide a testable local command entry point for manual extraction.
- `scripts/ingest_hourly_counts.py`
  - Thin wrapper that calls the CLI module.
- `src/urbanflow/ingestion/snapshots.py`
  - Add a generic immutable file snapshot helper for non-JSON snapshots.
- `src/urbanflow/ingestion/manifests.py`
  - Add optional structured metadata to the existing manifest writer so hourly manifests
    can record date range, selected columns, source dataset, query, and sensor filter.

The CLI will require either:

- `--year YYYY`, or
- both `--start-date YYYY-MM-DD` and `--end-date YYYY-MM-DD`.

There will be no unbounded default. This avoids accidental large downloads and makes
every formal extraction explicit.

## Data Flow

```text
CLI args / pipeline parameters
  -> date-range validation
  -> source where clause:
       sensing_date >= date'<start>' AND sensing_date <= date'<end>'
  -> records endpoint with limit=0 for expected source count
  -> exports/csv endpoint with selected columns and same where clause
  -> temporary CSV download file
  -> CSV row-count check
  -> immutable CSV snapshot after successful validation:
       data/raw/melbourne/hourly_counts/extracted_at=<YYYYMMDDTHHMMSSZ>/records.csv
  -> manifest:
       data/manifests/hourly_counts/<YYYYMMDDTHHMMSSZ>.json
```

The selected CSV columns will be ordered as:

```text
id,location_id,sensing_date,hourday,direction_1,direction_2,pedestriancount,sensor_name,location
```

The manifest will include the existing base fields plus hourly-count metadata:

- `source_dataset`
- `date_range.start`
- `date_range.end`
- `selected_columns`
- `source_where`
- `sensor_filter` set to `all`
- `snapshot_format` set to `csv`

## Error Handling

- Empty dataset names, invalid dates, missing date-range arguments, or `start_date >
  end_date` fail before any API request.
- A source count of zero fails before writing a snapshot.
- HTTP failures, invalid JSON from count queries, or unsuccessful CSV export requests
  raise `MelbourneApiError`.
- Existing snapshot or manifest paths still raise `FileExistsError`; ingestion should not
  overwrite immutable extraction artifacts.
- If the downloaded CSV cannot be parsed or its data-row count differs from the dynamic
  source count, ingestion raises an hourly-count export error before moving the file to
  its final immutable snapshot path and before writing a manifest.

## Testing Strategy

Unit tests stay network-free:

- Date-range parsing and validation, including `--year` expansion.
- Source `where` clause generation.
- API count-query behavior using mocked HTTP responses.
- CSV export snapshot writing with a fake API client.
- CSV row-count mismatch failure before manifest creation.
- CLI success summary and expected error exit codes.
- README/script help smoke test.

Manual live smoke testing will use a deliberately small explicit range, such as one
source day, before trying a full year. Generated raw data and manifests remain ignored by
Git.

## Out of Scope

This slice does not implement:

- Pandera validation rules.
- PostgreSQL, SQLAlchemy, or Alembic.
- Prefect orchestration.
- Sensor completeness ranking or top-10 sensor selection.
- Weather or holiday enrichment.
- Parquet output.
- Feature engineering, modeling, API serving, dashboarding, or monitoring.

Those belong to later planned delivery slices.

## Success Criteria

- A developer can run the hourly-count ingestion command for an explicit bounded date
  range and get a CSV snapshot plus manifest.
- The manifest records extraction time, source URL, source count, snapshot row count,
  date range, selected columns, and snapshot hash.
- The implementation avoids unbounded million-row defaults.
- Unit tests verify the behavior without network access.
- Full project Ruff checks, format checks, and pytest pass before integration.
