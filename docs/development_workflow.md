# Development Workflow

UrbanFlow AU uses a fast mainline workflow: keep `main` stable, do each task on a short-lived local branch, verify before integration, then push only `main`.

This workflow is based on three external conventions:

- [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow): keep changes small, branch from the main line, and integrate through reviewable units.
- [Trunk-Based Development](https://trunkbaseddevelopment.com/): prefer short-lived branches and frequent integration into the main line.
- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): make history easy to scan with commit prefixes such as `docs:`, `test:`, `feat:`, `fix:`, `build:`, and `ci:`.

## Branch and push policy

- `main` is the only shared branch for this project.
- Codex work happens on local branches named `codex/<small-slice>`.
- Codex branches are not pushed to GitHub.
- After a slice passes verification, merge the local Codex branch back into `main`.
- Push `main` after each verified merge when a GitHub remote is configured.
- If no remote is configured, keep the verified commits locally and report that pushing is blocked by the missing remote.

## Slice size

Each slice should be small enough to review and verify in one sitting. A good slice usually changes one project boundary, for example:

- repository workflow documentation;
- one ingestion endpoint parser;
- one snapshot writer plus manifest;
- one quality rule family;
- one API route group;
- one model baseline.

Avoid broad slices that mix ingestion, database migrations, modeling, and dashboard work. Those should be separate integrations.

## Implementation loop

For each slice:

1. Confirm the repository is clean on `main`.
2. Create an isolated local worktree on `codex/<small-slice>`.
3. Run the baseline quality gate before editing:
   ```powershell
   & .\.venv\Scripts\python.exe -m ruff check .
   & .\.venv\Scripts\python.exe -m ruff format --check .
   & .\.venv\Scripts\python.exe -m pytest
   ```
4. For behavior changes, use test-driven development:
   - write the failing test;
   - run it and confirm the expected failure;
   - implement the smallest passing code;
   - run the focused test;
   - run the full quality gate.
5. Commit intentionally with a Conventional Commit message.
6. Merge back to `main` only after fresh verification.
7. Re-run the quality gate on `main`.
8. Push `main` if `origin` exists.
9. Remove the local worktree and delete the local Codex branch after the merge is verified.

## Verification gate

Do not describe a slice as complete until these commands have just passed in the relevant worktree or on `main` after merge:

```powershell
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest
git status --short --branch
```

The expected state is zero Ruff issues, formatting already correct, all tests passing, and a clean Git status.

## Remote setup

This repository currently does not assume a GitHub remote. Add one only when the intended repository URL is known:

```powershell
git remote add origin <github-repository-url>
git push -u origin main
```

Never push `codex/*` branches unless the branch policy is explicitly changed.
