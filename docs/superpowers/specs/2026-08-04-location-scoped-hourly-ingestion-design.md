# Location-Scoped Hourly-Count Ingestion Design

## Context and decision

UrbanFlow AU already has a bounded City of Melbourne hourly-count CSV export,
immutable snapshot/manifest writing, a verified snapshot-to-supervised-CSV
bridge, local LightGBM artifact serving, and a Streamlit operations Dashboard.
The next practical step toward a real local serving demonstration is not another
Dashboard screen or a Docker bundle: it is a way to acquire a manageable,
reproducible historical slice for one explicitly chosen sensor.

The current hourly-count ingestion command filters only by date and always
records `"sensor_filter": "all"` in the manifest. Downloading five months for
all sensors would make the later direct 1–24-hour supervised expansion
unnecessarily large, while requiring an operator to manually modify a source
query would weaken provenance and reproducibility. The project requirements
also call for historical backfills to use official date and sensor filters.

This slice therefore adds an optional, positive-integer `location_id` filter to
the existing bounded ingestion path. It is deliberately not an automatic
"best sensor" selector: the operator owns the choice, and a later real-serving
runbook will verify the selected sensor has sufficient contiguous history.

## Goal

Allow an operator to create an immutable, manifest-verified hourly-count CSV
for one explicitly requested City of Melbourne `location_id` and a bounded
date range, using the same exact source filter for source counting and CSV
export.

For example:

```powershell
python scripts/ingest_hourly_counts.py `
  --start-date 2025-01-01 `
  --end-date 2025-05-31 `
  --location-id 101
```

The command remains an opt-in network operation. Unit tests use fake API
clients and never access Melbourne Open Data, PostgreSQL, MLflow, a model
artifact, FastAPI, or Streamlit.

## Considered approaches

### 1. Write only an operator runbook

This would be fastest to document, but the existing command would still export
all sensors for a five-month range. It leaves the main data-size and exact-query
provenance problem unresolved.

### 2. Add an explicit location-scoped ingestion option (selected)

Extend the current ingestion boundary with one optional `--location-id` value.
The existing source client already accepts a server-side `where` expression, so
the addition stays small, retains immutable snapshot semantics, and avoids
introducing a second downloader or a generic user-supplied query interface.

### 3. Add Docker Compose and automated data provisioning now

This would combine Windows database installation, network acquisition, database
writes, model training, serving, and UI evidence into one large failure domain.
It remains later work after the data path and real local demonstration are
proven.

## Scope

### Included

- Optional `--location-id POSITIVE_INTEGER` on
  `scripts/ingest_hourly_counts.py`.
- Domain validation for an optional location identifier so direct Python callers
  cannot bypass CLI validation.
- A deterministic source `where` clause that appends the selected location to
  the existing inclusive date conditions.
- Passing the same complete `where` clause to both `count_records` and
  `export_csv`.
- Provenance in the result JSON and manifest metadata describing whether the
  snapshot represents all sensors or one exact `location_id`.
- Offline unit tests and README examples for the scoped command.

### Explicitly excluded

- Downloading a real snapshot during tests or CI.
- Automatic sensor ranking, completeness scoring, gap repair, or choosing a
  "best" location.
- Changing the source dataset, CSV schema, selected columns, snapshot layout,
  manifest schema version, retry policy, or generic date-range behavior.
- PostgreSQL installation, migrations, database loading, model evaluation,
  artifact export, FastAPI configuration, Dashboard changes, Docker, or
  monitoring.

## Architecture

The existing pipeline remains the single owner of network I/O and snapshot
writing. The new filter is an explicit typed parameter that flows unchanged
from CLI to the shared query builder and then to both remote calls.

```text
--year OR --start-date/--end-date + optional --location-id
  -> date_range_from_args + positive location validation
  -> build_hourly_counts_where(date_range, location_id=...)
  -> count_records(dataset, where=exact_query)
  -> export_csv(dataset, where=the_same_exact_query)
  -> immutable records.csv + schema-v1 manifest
  -> existing validation / supervised CSV bridge / later local-serving runbook
```

No caller may supply raw Socrata SQL. The query builder formats only a validated
integer, preventing query-shape ambiguity and preserving a readable provenance
record.

## Public behavior and interfaces

### Domain helpers

`src/urbanflow/ingestion/hourly_counts.py` will expose:

```python
def validate_location_id(location_id: int | None) -> int | None:
    """Return None or a positive non-boolean City of Melbourne location id."""


def build_hourly_counts_where(
    date_range: HourlyCountDateRange,
    *,
    location_id: int | None = None,
) -> str:
    """Build the only permitted bounded hourly-count source query."""
```

With no location filter, the generated query remains byte-for-byte compatible
with the existing date-only query:

```text
sensing_date >= date'2025-01-01' AND sensing_date <= date'2025-05-31'
```

With `location_id=101`, it becomes:

```text
sensing_date >= date'2025-01-01' AND sensing_date <= date'2025-05-31' AND location_id = 101
```

`None`, an integer greater than zero, and only those values are valid. Boolean
values, zero, negative values, and non-integers raise `HourlyCountIngestionError`.

### Pipeline

`ingest_hourly_counts(...)` gains a keyword-only
`location_id: int | None = None`. It validates the argument before network I/O,
builds one `source_where` string, and passes that exact value to both source
operations. `HourlyCountIngestionResult` gains a `location_id: int | None`
field so a caller can distinguish an all-sensor snapshot from a scoped one.

The existing count equality check remains unchanged: the CSV row count must
equal the count returned for the same full query. An empty filtered result is a
hard error and creates no snapshot or manifest.

### Provenance

The schema-v1 manifest keeps all existing required fields and metadata. Its
`sensor_filter` value is:

- the existing string `"all"` when no location was supplied;
- `{"location_id": 101}` for a scoped snapshot.

`metadata.source_where` is the exact date-and-location query sent to the source.
The CLI success JSON adds `location_id`; it is an integer for a scoped run and
JSON `null` for the existing all-sensor behavior. No file is overwritten: the
existing timestamp-addressed snapshot and manifest collision handling remains
the authority.

### CLI

`--location-id` is independent of the existing date-range choice. It may be
used with `--year`, or with the required `--start-date`/`--end-date` pair. It
does not make date bounds optional and does not introduce a `--where` escape
hatch.

Malformed location arguments remain an argparse usage error with exit code `2`.
Source, export, row-count, snapshot, or manifest failures retain the existing
stderr `error: ...` and exit code `1` behavior.

## Testing and acceptance

All new tests are local and deterministic.

1. Domain tests prove the exact all-sensor query remains unchanged, a valid
   location produces the exact appended predicate, and invalid location values
   are rejected.
2. Pipeline tests use a fake source client to prove `count_records` and
   `export_csv` receive the same scoped query, result metadata preserves the
   location, and manifest metadata records the structured filter and exact
   query.
3. Existing no-filter pipeline tests continue to assert `sensor_filter ==
   "all"` and the original date-only query.
4. CLI tests cover a successful scoped invocation, `--year` combined with a
   location, invalid/zero/negative identifiers, and script help text.
5. The focused ingestion suites, repository Ruff check, formatter check, full
   pytest suite, and `scripts/ingest_hourly_counts.py --help` all pass without
   network, PostgreSQL, MLflow, model, or Dashboard dependencies.

## Operator handoff after this slice

The later real-serving demonstration will use the new bounded location filter
to acquire a historical interval of at least five complete months for an
operator-selected sensor. The operator will then validate the snapshot, confirm
the chosen sensor has at least 168 contiguous observed hours at its cutoff,
create an explicit Victorian holiday calendar, build the supervised CSV,
evaluate/export the existing LightGBM artifact, load the validated records into
an operator-owned PostgreSQL instance, and configure FastAPI plus Streamlit.

Historical source data remains historical: no `URBANFLOW_API_MAX_DATA_AGE_HOURS`
threshold will be set merely to make `/health` report fresh. A later real
forecast may therefore coexist with a truthful degraded freshness state.
