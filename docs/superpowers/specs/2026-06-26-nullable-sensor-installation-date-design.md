# Nullable Sensor Installation Date Design

## Purpose

Make sensor-location ingestion compatible with the current City of Melbourne
source contract without losing a valid sensor record. A live fetch on 2026-06-26
returned 136 records; location `42` (`UM1_T`) had `installation_date: null`
while every location-identifying and coordinate field was present.

## Selected Contract

`installation_date` is optional descriptive metadata in the normalized sensor
location record:

```python
installation_date: str | None
```

The parser will preserve a source `null` as JSON `null`. It will not invent a
date, drop the record, or fail the whole snapshot for this missing metadata.

The following fields remain strict requirements because they identify or locate
the sensor: `location_id`, `sensor_description`, `sensor_name`, `status`,
`latitude`, and `longitude`.

## Data Flow and Error Handling

```text
source record with installation_date=null
  -> parse_sensor_location
  -> SensorLocation(installation_date=None)
  -> normalized JSON with installation_date=null
  -> unchanged snapshot and manifest flow
```

Missing or invalid strict fields continue to raise `SensorLocationParseError`
before the pipeline writes an output. A future data-quality layer may report
missing installation dates, but this ingestion slice only preserves the source
truth and keeps the location dataset complete.

## Testing

Tests will first show that a record with `installation_date=None` fails under
the old contract. The minimal implementation then changes the dataclass type,
removes this field from the strict required-field list, and uses the existing
optional-string coercion path.

The test suite will verify that normalization returns `None` for the date while
still rejecting an invalid coordinate. The existing runner tests stay
network-free. A fresh live runner command will confirm the official 136-record
response can produce a snapshot and manifest.

## Scope Boundaries

This change does not relax any coordinate, ID, name, or status validation. It
does not add a data-quality reporting framework, alter manifest structure, or
change the hourly-count ingestion design.
