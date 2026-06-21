# UrbanFlow AU Project Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal Python 3.11-compatible package foundation with reproducible local quality checks and matching GitHub Actions CI.

**Architecture:** Use a `src` package layout with package metadata and all tool configuration in `pyproject.toml`. Keep the first executable surface to a versioned package import, verified by pytest; keep CI and local commands identical and leave data/MLOps dependencies for later vertical slices.

**Tech Stack:** Python 3.11+, setuptools, pytest, Ruff, GitHub Actions

---

### Task 1: Repository hygiene and project overview

**Files:**
- Modify: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Add repository exclusions**

```gitignore
# Local worktrees
.worktrees/

# Python
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.venv/

# Test and quality caches
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/

# Secrets and local configuration
.env
.env.*
!.env.example

# Editors and operating systems
.idea/
.vscode/
.DS_Store
Thumbs.db

# Local data and generated artifacts
data/raw/
data/interim/
data/processed/
data/cache/
models/*
!models/.gitkeep
mlruns/
```

- [ ] **Step 2: Add the honest foundation README**

````markdown
# UrbanFlow AU

UrbanFlow AU is an end-to-end platform for forecasting hourly pedestrian demand at selected City of Melbourne sensor locations. It will connect reproducible public-data ingestion, leakage-safe time-series evaluation, model serving, an operations dashboard, and MLOps monitoring.

> **Project status:** foundation stage. The data pipeline and measured model results have not been implemented yet; no performance claims are made.

## Requirements

- Python 3.11 (CI reference version)
- Git

The complete product scope is documented in [urbanflow-au_requirements.md](urbanflow-au_requirements.md). The foundation design is in [docs/superpowers/specs/2026-06-20-project-foundation-design.md](docs/superpowers/specs/2026-06-20-project-foundation-design.md).

## Local development

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the same quality checks used by CI:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

## Planned delivery slices

1. Melbourne sensor and hourly-count ingestion with immutable snapshots and manifests.
2. Data validation, PostgreSQL persistence, and Prefect orchestration.
3. Leakage-safe features, rolling-origin backtests, and MLflow tracking.
4. FastAPI forecasts, Streamlit operations views, and Evidently monitoring.
5. Docker Compose packaging, evaluation evidence, screenshots, and portfolio documentation.

## Data policy

The repository will contain only small deterministic fixtures, sample data, and manifests. Full raw data, secrets, model artifacts, and local experiment stores remain untracked.
````

- [ ] **Step 3: Verify Git ignores the isolated-worktree directory**

Run: `git check-ignore .worktrees`
Expected: `.worktrees`

- [ ] **Step 4: Commit repository hygiene and overview**

```powershell
git add .gitignore README.md
git commit -m "chore: add repository foundation files"
```

### Task 2: Package import and version contract

**Files:**
- Create: `tests/unit/test_package.py`
- Create: `src/urbanflow/__init__.py`
- Create: `pyproject.toml`

- [ ] **Step 1: Write the failing package test**

```python
from importlib.metadata import version

import urbanflow


def test_package_exposes_installed_version() -> None:
    assert urbanflow.__version__ == version("urbanflow-au")
```

- [ ] **Step 2: Run the test to verify RED**

Run: `python -m pytest tests/unit/test_package.py -v`
Expected: collection fails with `ModuleNotFoundError: No module named 'urbanflow'` because the package does not exist yet.

- [ ] **Step 3: Add package metadata and tool configuration**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "urbanflow-au"
dynamic = ["version"]
description = "Melbourne pedestrian demand forecasting and MLOps platform"
readme = "README.md"
requires-python = ">=3.11"
authors = [{ name = "UrbanFlow AU contributors" }]
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=8.3,<10",
    "ruff>=0.11,<1",
]

[tool.setuptools.dynamic]
version = { attr = "urbanflow.__version__" }

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-ra --strict-config --strict-markers"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "UP", "B"]
```

- [ ] **Step 4: Add the minimal package implementation**

```python
"""UrbanFlow AU package."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Install the package in an isolated environment**

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Expected: editable installation completes with `urbanflow-au`, pytest, and Ruff installed.

- [ ] **Step 6: Run the test to verify GREEN**

Run: `& .\.venv\Scripts\python.exe -m pytest tests/unit/test_package.py -v`
Expected: `1 passed`.

- [ ] **Step 7: Commit the package contract**

```powershell
git add pyproject.toml src/urbanflow/__init__.py tests/unit/test_package.py
git commit -m "build: add tested Python package foundation"
```

### Task 3: Continuous integration

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Add GitHub Actions CI**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  quality:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install project
        run: python -m pip install -e ".[dev]"

      - name: Lint
        run: python -m ruff check .

      - name: Check formatting
        run: python -m ruff format --check .

      - name: Test
        run: python -m pytest
```

- [ ] **Step 2: Run the complete local quality gate**

```powershell
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
```

Expected: Ruff reports no issues, formatting check reports all files formatted, and pytest reports `1 passed`.

- [ ] **Step 3: Commit CI**

```powershell
git add .github/workflows/ci.yml
git commit -m "ci: add foundation quality gate"
```

### Task 4: Final foundation verification

**Files:**
- Verify: all tracked files

- [ ] **Step 1: Reinstall the current package metadata**

Run: `& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"`
Expected: editable installation succeeds.

- [ ] **Step 2: Run lint, formatting, and tests from repository root**

```powershell
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
```

Expected: all commands exit with status 0 and pytest reports `1 passed`.

- [ ] **Step 3: Inspect repository state**

Run: `git status --short --branch`
Expected: the feature branch is clean and contains no `.venv`, cache, data, secret, or generated artifact entries.
