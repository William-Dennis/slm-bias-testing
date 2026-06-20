# SLM Bias Testing 🔍

**Bias benchmarks for small language models (<1B params).**

Track how bias changes as models get smaller, newer, and smarter.
Run your own evaluations, compare models, and visualise trends.

```bash
uv sync --extra dev
uv run python scripts/run_benchmarks.py --models smollm2-135m --benchmark all --pool-size 4
```

---

## What this does

Four bias benchmarks, one command per model:

| Benchmark | What it measures | Samples |
|---|---|---|
| **StereoSet** | Stereotype score across gender, race, religion, profession | 2106 |
| **WinoBias** | Gender pronoun resolution bias (pro vs anti-stereotypical) | 1584 |
| **CV Screening** | Scoring bias by name, gender, ethnicity, university prestige | 600 CVs × 10 runs |
| **Demographic Bias** | Output length disparity across 8 demographic groups | 400 prompts |

---

## Models (under 1B params)

10 models across 5 families, spanning July 2024 to October 2025:

| Name | Ollama Tag | Params | Release | Family |
|---|---|---|---|---|
| smollm-135m | smollm:135m | 135M | 2024-07 | huggingface |
| smollm-360m | smollm:360m | 360M | 2024-07 | huggingface |
| qwen25-05b | qwen2.5:0.5b | 500M | 2024-09 | alibaba |
| smollm2-135m | smollm2:135m | 135M | 2024-11 | huggingface |
| smollm2-360m | smollm2:360m | 360M | 2024-11 | huggingface |
| gemma3-270m | gemma3:270m | 270M | 2025-03 | google |
| qwen3-06b | qwen3:0.6b | 600M | 2025-04 | alibaba |
| lfm2-350m | sam860/lfm2:350m | 350M | 2025-07 | liquid |
| lfm2-700m | sam860/lfm2:700m | 700M | 2025-07 | liquid |
| granite4-350m | granite4:350m | 350M | 2025-10 | ibm |

---

## Quick start

```bash
# Install
uv sync --extra dev

# Run all benchmarks on one model (uses Node.js pool for parallelism)
uv run python scripts/run_benchmarks.py --models smollm2-135m --benchmark all --pool-size 4

# Run a specific benchmark with limited samples
uv run python scripts/run_benchmarks.py --models smollm2-135m --benchmark stereoset --max-samples 20 --pool-size 2

# Run all models, all benchmarks
uv run python scripts/run_benchmarks.py --models all --pool-size 6

# Temporal trend analysis
uv run python -m slm_bias_testing.temporal
```

---

## Outputs

```
results/
  {model}/
    {benchmark}/
      results.json       — Summary scores
      {benchmark}.json   — Full per-item results
      plots/             — Violin plots (CV screening)
  analysis_summary.txt   — Statistical analysis (group means, CI, Cohen's d)

figs/
  temporal_trends.png         — Bias score vs release date
  family_comparison.png       — Per-family bias comparison
```

---

## Project structure

```
src/slm_bias_testing/
  registry.py           — Model definitions (name → ollama tag)
  benchmark_runner.py   — Core runner with pool lifecycle
  cv_screening.py       — CV screening benchmark
  model_clients.py      — OllamaPoolClient (Node.js pool subprocess)
  temporal.py           — Temporal analysis & trend plots
  analysis.py           — Statistical helpers (CI, Cohen's d, variance)
  benchmarks/
    stereoset.py         — StereoSet benchmark
    winobias.py          — WinoBias gender coreference benchmark
    demographic_bias.py  — Output length disparity benchmark

scripts/
  run_benchmarks.py     — CLI entry point (one or all models, all benchmarks)
  ollama_pool.mjs       — Node.js worker pool for parallel Ollama calls

examples/             — CV data, job description, templates
tests/                — 114 tests
```

---

## Prerequisites

- [Ollama](https://ollama.ai) — all models run locally
- [Node.js](https://nodejs.org) — for the parallel worker pool
- `uv` (or `pip`) for Python dependencies

---

## Tests

```bash
uv run pytest tests/ -q
uv run ruff check src tests
```

---

*Questions? Open an issue or ping @William-Dennis.*
