# Runtime Readiness Serving E2E Final Fix Report

Date: 2026-07-30

Branch: `codex/runtime-readiness-serving-e2e`

Worktree: `D:\Github项目\UrbanFlow-AU\.worktrees\runtime-readiness-serving-e2e`

## Status

All four final-review findings were fixed in one focused, offline change wave:

1. Health polling and Dashboard API requests now use smoke-owned
   `httpx.Client` instances with `trust_env=False`. The polling client is
   context-managed per poll, and the Dashboard smoke client closes its owned
   HTTP client through the existing smoke cleanup path.
2. Process cleanup now attempts `kill()` followed by a bounded `wait()` when
   the process remains alive after an unexpected `terminate()` or non-timeout
   `wait()` exception. It retains a safe `RuntimeError` outcome so the caller
   records cleanup failure while its remaining cleanup steps continue.
3. An `OverflowError` raised while converting a canonical positive integer to
   `timedelta` is translated to `ApiRuntimeConfigError`.
4. README and the approved plan now state that the smoke makes no external
   network request and that all smoke HTTP stays on loopback.

No real PostgreSQL, Uvicorn, Streamlit, HTTP server, or manual serving smoke was
started.

## TDD Evidence

### RED

Command:

```powershell
$env:PYTHONPATH='D:\Github项目\UrbanFlow-AU\.worktrees\runtime-readiness-serving-e2e\src'
& 'D:\Github项目\UrbanFlow-AU\.venv\Scripts\python.exe' -m pytest tests/unit/api/test_serving_e2e_smoke.py tests/unit/api/test_health.py -q
```

Expected result observed: `5 failed, 36 passed in 13.97s`.

The five failures independently demonstrated:

- health polling still called module-level `httpx.get`;
- the proxy-independent owned Dashboard smoke client did not exist;
- `terminate()` exceptions escaped before a hard stop;
- non-timeout `wait()` exceptions escaped before a hard stop;
- oversized canonical positive integers exposed raw `OverflowError`.

### GREEN

The same focused command passed after the minimal implementation:

```text
41 passed in 2.73s
```

After Ruff formatting, it was rerun and remained green:

```text
41 passed in 3.38s
```

Focused Ruff verification:

```text
python -m ruff check <four focused source/test files>
All checks passed!

python -m ruff format --check <four focused source/test files>
4 files already formatted
```

## Full Quality Gate

Each gate was run as a separate command with the root interpreter and
`PYTHONPATH` pointing at this worktree's `src`.

```text
python -m ruff check .
All checks passed!
```

```text
python -m ruff format --check .
173 files already formatted
```

One standalone bounded full test run was executed with a 240-second limit:

```text
python -m pytest -q
590 passed, 238 warnings in 123.20s (0:02:03)
```

The warnings are existing sklearn, joblib, NumPy/pandas deprecation and
imputation warnings; warning cleanup was outside this fix scope. The bounded
run completed, so it was not retried.

`git diff --check` passed. The manual configured-serving smoke was intentionally
not run.

## Files Changed

Source:

- `src/urbanflow/api/serving_e2e_smoke.py`
- `src/urbanflow/api/services.py`

Tests:

- `tests/unit/api/test_serving_e2e_smoke.py`
- `tests/unit/api/test_health.py`

Documentation:

- `README.md`
- `docs/superpowers/plans/2026-07-30-runtime-readiness-serving-e2e.md`
- `.superpowers/sdd/2026-07-30-runtime-readiness-serving-e2e/final-fix-report.md`

## Self-Review

- Both loopback HTTP paths explicitly disable proxy/environment trust.
- Every newly owned HTTP client has an explicit close path.
- Existing `http_poller` and `client_factory` test-injection seams are
  unchanged.
- Normal terminate/wait behavior and timeout-triggered kill behavior are
  preserved.
- Unexpected soft-stop exceptions cause a hard-stop attempt only while the
  process still reports alive, then produce a generic safe cleanup error.
- Later log, schema, engine, and temporary-directory cleanup remains in the
  existing independent `finally` blocks.
- Public API routes, response schemas, health semantics, database fallback
  policy, schema isolation, and secret redaction were not changed.
- No dependency, CI, Docker, migration, page, or environment change was added.

## Concerns

No blocking concerns. Full pytest still reports 238 pre-existing warnings; they
were recorded but not modified. Runtime integration was deliberately not
exercised because this final-fix task required offline verification only.
