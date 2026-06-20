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
├── benchmark.py         # Core CV screening benchmark logic
├── call_api.py          # Ollama + Transformers model clients (OllamaClient, Model)
├── ollama_setup.py      # Ollama server lifecycle management
├── registry.py          # Model registry (name → metadata)
├── runner.py            # CLI runner for per-model benchmarks
├── temporal.py          # Temporal bias trend analysis + plotting
├── transformers.py      # HuggingFace Transformers model wrapper
└── benchmarks/
    ├── __init__.py      # BaseBenchmark ABC
    ├── demographic_bias.py
    ├── stereoset.py
    └── winobias.py
```

## Key Patterns

- **No mutable globals.** Ollama state lives in `OllamaClient` instances, not module vars.
- **Dependency injection.** `Model` accepts optional `OllamaClient`. Tests mock at the client level.
- **Type annotations required.** Mypy strict mode (`disallow_untyped_defs`).
- **All benchmarks extend `BaseBenchmark`.** Must implement `load_dataset()` and `evaluate()`.

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
