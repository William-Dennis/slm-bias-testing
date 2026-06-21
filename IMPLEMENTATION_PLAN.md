# Implementation Plan: Node.js Ollama Pool Manager

**Issue:** #30 — Replace Python threading with Node.js worker pool for parallel benchmarking.

## Problem

Python-side `ThreadPoolExecutor` only works for cv-screening. StereoSet, WinoBias, and demographic-bias run sequentially — 1 request at a time. ~33 calls/min, ~40h for a full 10-model run.

## Solution

Replace Python threading with a Node.js worker pool (`scripts/ollama_pool.mjs`). Python stays sequential and simple. Node.js handles all Ollama API calls.

```
Python (benchmark logic)          Node.js (ollama_pool.mjs)
─────────────────────             ─────────────────────────
1. Generate jobs (batch of 40)    1. Read JSONL from stdin
2. Write JSONL to stdin ──────►  2. Dispatch to worker pool
3. Read results from stdout ◄──  3. POST /api/chat to Ollama
4. Checkpoint each result        4. Write JSONL to stdout
```

## Decisions

| # | Decision |
|---|---|
| 1 | Node.js pool as standalone tool, lives in `scripts/` |
| 2 | Model-aware batch dispatch (groups jobs by same model to avoid GPU thrashing) |
| 3 | Batch size configurable, default 40. Results stream back as completed, checkpoint each item |
| 4 | 3 attempts per job in pool (1s/2s/4s backoff), then permanently fail with warning |
| 5 | Always-batch mode when pool available, no `OllamaModel` fallback |
| 6 | Pool kills/restarts Ollama on startup with correct `OLLAMA_NUM_PARALLEL` |
| 7 | `keep_alive=30s` default |
| 8 | One model at a time, no model-level parallelism |
| 9 | Pool stays alive across all 4 benchmarks per model, restarted when switching models |
| 10 | Checkpointing stays in benchmark code, not in `OllamaPoolClient` |

## Files to Create

| File | Description |
|---|---|
| `scripts/ollama_pool.mjs` | Node.js worker pool manager |
| `scripts/run_benchmarks.py` | Thin CLI wrapper |
| `src/slm_bias_testing/model_clients.py` | `OllamaPoolClient` class |
| `src/slm_bias_testing/cv_screening.py` | CV screening benchmark (renamed from `benchmark.py`) |
| `src/slm_bias_testing/benchmark_runner.py` | Core runner with pool lifecycle (renamed from `runner.py`) |
| `tests/test_model_clients.py` | `OllamaPoolClient` unit tests |
| `tests/test_ollama_pool.py` | Node.js pool integration tests |

## Files to Delete

| File | Reason |
|---|---|
| `src/slm_bias_testing/call_api.py` | Replaced by `model_clients.py` |
| `src/slm_bias_testing/ollama_setup.py` | Pool owns Ollama lifecycle |
| `src/slm_bias_testing/benchmark.py` | Renamed to `cv_screening.py` |
| `src/slm_bias_testing/runner.py` | Renamed to `benchmark_runner.py` |
| `scripts/run_single_model.py` | Merged into `run_benchmarks.py` |
| `scripts/run_parallel.py` | Model-level parallelism removed |
| `scripts/run_experiments.py` | Merged into `run_benchmarks.py` |
| `tests/test_call_api.py` | Tests deleted classes |

## Files to Modify

| File | Changes |
|---|---|
| `src/slm_bias_testing/benchmarks/__init__.py` | Add `pool_client` param to `evaluate()` |
| `src/slm_bias_testing/benchmarks/stereoset.py` | Refactor to always-batch with `predict_batch()` |
| `src/slm_bias_testing/benchmarks/winobias.py` | Same refactor |
| `src/slm_bias_testing/benchmarks/demographic_bias.py` | Same refactor |
| `src/slm_bias_testing/temporal.py` | Update help string reference |
| `docs/ollama-pool-manager.md` | Update to reflect decisions |
| `README.md` | Update script names and usage |
| `tests/test_benchmarks.py` | `MockModel` → `MockPoolClient`, pool-mode tests |
| `tests/test_runner.py` | Adjust for `benchmark_runner.py` |
| `tests/test_main.py` | Update imports |

## Execution Order

1. Create `scripts/ollama_pool.mjs` — standalone, testable
2. Create `model_clients.py` — `OllamaPoolClient`
3. Refactor `benchmark.py` → `cv_screening.py` — remove threading, pool batching
4. Refactor `runner.py` → `benchmark_runner.py` — pool lifecycle, model loop
5. Refactor benchmarks (stereoset, winobias, demographic_bias) — always-batch
6. Create `scripts/run_benchmarks.py` — thin CLI
7. Delete old files
8. Write new + update existing tests
9. Update docs (README, design doc, temporal.py)
10. Run CI checks

## Node.js Pool Protocol

### Job Format (Python → Node.js, stdin JSONL)
```json
{"id": "stereoset_042_stereotype", "model": "smollm:135m", "prompt": "...", "temperature": 0.0, "num_ctx": 2048, "keep_alive": 30}
```

### Result Format (Node.js → Python, stdout JSONL)
```json
{"id": "stereoset_042_stereotype", "response": "85", "error": null, "latency_ms": 1234}
```

### Flow Control
- Python writes all jobs in batch, reads results as they stream back
- Results arrive out of order (matched by `id`)
- Python checkpoints each result immediately
- After reading N results (batch size), Python starts next batch
- After all benchmarks: Python closes stdin → pool drains → exits

## Pool CLI Flags
```
--max-pool <n>          Upper bound on concurrent workers (default: 6)
--min-pool <n>          Lower bound (default: 1)
--adaptive              Enable adaptive concurrency (default: true)
--ram-floor <gb>        Minimum free RAM (default: 2)
--latency-ceiling <ms>  Pause if avg latency exceeds this (default: 15000)
--timeout <ms>          Per-request timeout (default: 30000)
--retries <n>           Retry count per request (default: 2)
--ollama-host <url>     Ollama base URL (default: http://localhost:11434)
```

## Python CLI
```bash
uv run python scripts/run_benchmarks.py --models smollm-135m,gemma3-270m --pool-size 4 --batch-size 40
uv run python scripts/run_benchmarks.py --models all --pool-size 6
```
