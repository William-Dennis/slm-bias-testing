# AGENTS.md — SLM Bias Testing

## Project Overview

Bias evaluation benchmarks for small language models. Tests LLMs on
demographic bias (CV screening, StereoSet, WinoBias, demographic completion).

## Repository Rules (non-negotiable)

1. **Never commit directly to `main`.** Branch → PR → review → merge.
2. **One PR per issue.** Link with `Closes #N`.
3. **All CI must pass** before merge: Lint, Type Check, Test (3.11 + 3.12).
4. **No approving review required** (solo repo). All threads must be resolved.
5. **All review threads resolved** before merge.
6. **Squash merge only.** Linear history enforced.
7. **No force push** to any shared branch.
8. **No admin merge or bypass** of branch protection.

## Workflow

- Issue first, then branch, then PR.
- Branch naming: `feat/`, `fix/`, `refactor/`, `docs/`, `test/`.
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
- Run `uv run ruff check src tests && uv run ruff format --check src tests && uv run ty check && uv run pytest` before pushing.

## Architecture

```
src/slm_bias_testing/
├── analysis.py          # Statistical analysis (Cohen's d, variance breakdown)
├── benchmark_runner.py  # Core runner with pool lifecycle, model iteration
├── cv_screening.py      # CV screening benchmark (batched via pool)
├── model_clients.py     # OllamaPoolClient (Node.js pool subprocess)
├── registry.py          # Model registry (name → metadata)
├── temporal.py          # Temporal bias trend analysis + plotting
└── benchmarks/
    ├── __init__.py      # BaseBenchmark ABC (pool_client param)
    ├── demographic_bias.py
    ├── stereoset.py
    └── winobias.py

scripts/
├── run_benchmarks.py    # CLI entry point (one or all models, all benchmarks)
└── ollama_pool.mjs      # Node.js worker pool for parallel Ollama calls
```

## Key Patterns

- **Node.js pool for parallelism.** Python stays sequential. Pool handles all Ollama API calls.
- **Batch protocol.** Python writes JSONL jobs to pool stdin, reads JSONL results from stdout.
- **Per-model pool lifecycle.** Pool kills/restarts Ollama with correct OLLAMA_NUM_PARALLEL per model.
- **Checkpoint after every batch.** Batch size configurable (default 40), checkpoint each item as result arrives.
- **No mutable globals.** Pool state lives in subprocess, not Python module vars.
- **Type annotations required.** Ty strict mode.
- **All benchmarks extend `BaseBenchmark`.** Must implement `load_dataset()` and `evaluate()`.
- **Benchmarks use `pool_client.predict_batch()`** for parallel processing when pool is available.

## CI Pipeline

`.github/workflows/ci.yml` runs on push to main and all PRs:
- Lint: `ruff check` + `ruff format --check`
- Type Check: `ty check`
- Test: `pytest` on Python 3.11 + 3.12 matrix

Branch protection requires all 4 checks to pass with strict mode (branch must be up-to-date with main).

## Test Conventions

- Tests in `tests/` mirroring `src/` structure.
- Mark slow/integration tests: `@pytest.mark.slow`, `@pytest.mark.integration`.
- CI runs: `pytest -m "not integration and not slow"`.
- Mock external services (Ollama, HuggingFace) — no network calls in unit tests.
