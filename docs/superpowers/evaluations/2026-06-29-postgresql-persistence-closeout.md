# PostgreSQL persistence closeout evaluation

Date: 2026-06-29

## Decision

Add a real, manually triggered local PostgreSQL smoke test before closing the
PostgreSQL persistence stage.

The smoke test should not run as part of the default unit-test suite or default
CI path. It requires an explicit PostgreSQL URL through
`URBANFLOW_SMOKE_DATABASE_URL` or `--database-url`.

## Rationale

The persistence stage already has unit coverage for configuration, SQLAlchemy
models, PostgreSQL upsert statement generation, row transformation, repository
calls, loader validation gates, and CLI behavior. Those tests make the
application logic cheap to validate, but they do not prove that a live
PostgreSQL connection can create the schema, execute the PostgreSQL dialect
upserts, satisfy the foreign key, and read inserted rows back in one real
database transaction.

A local smoke test closes that last integration gap without making every
developer or CI run provision PostgreSQL.

## Scope

The smoke test:

- creates a temporary schema in the target database;
- creates the current SQLAlchemy metadata inside that schema;
- upserts one synthetic sensor row and one synthetic hourly pedestrian-count row;
- reads the row counts back from PostgreSQL;
- drops the temporary schema in cleanup.

The smoke test intentionally does not:

- download Melbourne data;
- run Alembic migrations;
- start Docker or install PostgreSQL;
- run automatically when `pytest` runs.

## Closeout result

PostgreSQL persistence is considered closed when:

- the manual smoke path exists and is documented;
- its CLI behavior is covered by unit tests that do not require PostgreSQL;
- default quality gates still pass.

## Current verification note

The manual smoke runner is implemented and documented. It was not executed
against a live PostgreSQL database in this closeout pass because
`URBANFLOW_SMOKE_DATABASE_URL` was not set in the local environment.
