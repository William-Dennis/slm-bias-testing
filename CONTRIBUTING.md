# Contributing to SLM Bias Testing

## Development Setup

```bash
git clone git@github.com:William-Dennis/slm-bias-testing.git
cd slm-bias-testing
uv sync --extra dev
```

## Workflow

This repo enforces an **issue → branch → PR → review → merge** model.
**Never commit directly to `main`.** No exceptions.

1. Create an issue describing the change.
2. Create a branch from `main`:
   ```bash
   git checkout main && git pull
   git checkout -b feat/issue-N-description
   ```
3. Make changes. Write tests first (TDD), then implement.
4. Run checks locally before pushing (same commands as CI):
   ```bash
   uv run ruff check src tests scripts
   uv run ruff format --check src tests scripts
   uv run ty check
   uv run pytest -m "not integration and not slow"
   node --check scripts/ollama_pool.mjs
   ```
5. Push and open a PR targeting `main`. Link the issue with `Closes #N`.
6. All CI checks must pass (Lint, Type Check, Test 3.11, Test 3.12).
7. No approving review required (solo repo) — but no admin bypass.
   Self-review the diff before merge; there are no automated review
   threads (CodeRabbit was removed).
8. Squash merge. Branch auto-deleted after merge.

## Branch Protection (main)

- **Strict status checks:** Lint, Type Check, Test (Python 3.11), Test (Python 3.12)
- **Required approving reviews:** 0 (solo repo)
- **Dismiss stale reviews:** yes (new pushes invalidate old approvals)
- **Conversation resolution:** all threads must be resolved
- **Linear history:** squash merges only (no merge commits)
- **Enforce admins:** yes (no bypassing protection)
- **No force push, no branch deletion**

## Code Quality Standards

- **Linting:** Ruff (E, W, F, I, UP, B, SIM, TCH, RUF)
- **Formatting:** Ruff formatter (line length 100)
- **Type checking:** Ty
- **Testing:** Pytest. All new code must have tests.
- **SOLID:** Single responsibility per module/class. No mutable global state.
- **No mutable globals.** Use dependency injection (classes, not module-level state).

## Commit Convention

```
type: concise subject line

Optional body.
```

Types: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`

## Merge Strategy

**Squash merge** for all feature branches. One PR per issue.
Branches auto-delete after merge.

## Running Benchmarks

Benchmarks require Ollama (and Node.js for the parallel pool):

```bash
ollama serve &
# one model, one benchmark
uv run python scripts/run_benchmarks.py --models gemma3-1b --benchmark cv-screening
# every registered model, every benchmark
uv run python scripts/run_benchmarks.py --models all --benchmark all --pool-size 4
```
