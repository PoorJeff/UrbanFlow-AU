# UrbanFlow AU Project Foundation Design

## Purpose

Create a small, reproducible Python foundation for UrbanFlow AU before implementing data ingestion. The foundation will make every later vertical slice installable, linted, and testable in the same way locally and in GitHub Actions.

This design covers repository infrastructure only. It does not implement API access, data storage, feature engineering, modeling, services, dashboards, orchestration, or deployment.

## Considered Approaches

### 1. Scaffold the complete target repository immediately

This would create every directory shown in the project requirements, plus placeholder applications and configuration. It makes the final shape visible early, but produces many empty modules whose interfaces have not yet been designed.

### 2. Build a minimal verified foundation first (selected)

Create only the packaging, quality tooling, CI workflow, package entry point, and one package smoke test. New directories will then be added by working vertical slice rather than in anticipation of future work.

This approach is selected because it establishes engineering standards without locking the project into premature module boundaries.

### 3. Start directly with the Melbourne API ingestion slice

This delivers user-visible behavior sooner, but mixes repository-tooling decisions with HTTP pagination, retries, schemas, caching, and manifests. That makes the first change unnecessarily broad and harder to diagnose.

## Foundation Structure

The implementation will add these files:

- `pyproject.toml`: Python 3.11 package metadata, editable installation, pytest settings, Ruff rules, and development dependencies.
- `src/urbanflow/__init__.py`: the package boundary and a single source for the initial package version.
- `tests/unit/test_package.py`: a smoke test proving the installed package can be imported and exposes its version.
- `.github/workflows/ci.yml`: a small CI job that installs the project, runs Ruff checks, checks formatting, and runs pytest on Python 3.11.
- `.gitignore`: Python, test, environment, IDE, local data, model artifact, and secret exclusions.
- `README.md`: a concise project definition and development commands; measured model results and screenshots will only be added when real outputs exist.

The existing `urbanflow-au_requirements.md` remains the authoritative product requirements document.

## Tooling Decisions

- Use a standard `src` package layout so tests exercise the installed package instead of importing accidentally from the repository root.
- Support `python -m pip install -e ".[dev]"`; a separate package-manager requirement and lock file are deferred until the dependency set becomes meaningful.
- Configure Ruff as both linter and formatter to keep local and CI commands identical.
- Use pytest for all test layers. Only the unit-test directory is created in this slice; integration fixtures are added alongside the first integration boundary.
- Run CI on Python 3.11, matching the project requirements. A version matrix is unnecessary for this portfolio application.

## Verification and Failure Behaviour

The foundation is accepted only when all of the following commands succeed from a clean project environment:

```powershell
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

CI will run the same checks and fail on the first unsuccessful command. No check will download Melbourne data or require PostgreSQL, external services, credentials, or network access after dependency installation.

## Git Workflow

The local repository uses `main` as its primary branch. Work may later be developed on local Codex branches, but only changes merged into `main` will be pushed to GitHub. No remote repository will be created or guessed; a remote is added only after its intended URL and visibility are known.

## Success Criteria

- The repository is on `main` and contains the requirements and this design under version control.
- The package installs in editable mode on Python 3.11.
- The package smoke test passes.
- Ruff lint and formatting checks pass locally and use the same configuration in CI.
- Local secrets, large data, model artifacts, and generated caches are excluded from Git.
- No empty subsystem scaffolding is created.

## Next Slice

After this foundation is approved and implemented, the first functional design will cover sensor-location ingestion as a narrow vertical slice: HTTP client behaviour, bounded retries, response validation, pagination, immutable snapshot output, manifest generation, and deterministic tests using fixed responses.
