# SLM Bias Testing 🔍

**Bias benchmarks for small language models (<1B params).**

Track how bias changes as models get smaller, newer, and smarter.
Run your own evaluations, compare models, and visualise trends.

```bash
uv sync --extra dev
uv run python -m slm_bias_testing.runner smollm2-135m --benchmark all
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

## Main results

Results so far (smoke runs, `--max-samples 20` unless noted). Charts are committed under [`figs/`](figs/) and regenerate with `uv run python -m slm_bias_testing.visualisations` / `uv run python -m slm_bias_testing.temporal`.

### StereoScore vs release date (lower = less stereotyped)

<img src="figs/temporal_trends.png" width="700" alt="Bias score vs release date">

| Model | Family | Release | StereoScore |
|---|---|---|---|
| smollm2-135m | huggingface | 2024-11 | **5.0%** |
| gemma3-270m | google | 2025-03 | **10.0%** |
| smollm-135m | huggingface | 2024-07 | 20.0% |
| qwen25-15b | alibaba | 2024-09 | 20.0% |
| smollm-360m | huggingface | 2024-07 | 25.0% |
| smollm2-360m | huggingface | 2024-11 | 35.0% |
| qwen25-05b | alibaba | 2024-09 | 35.0% |

**Headline finding:** bias does not fall monotonically with release date — the best score (5%) comes from a 2024-11 model, while newer models sit at 10–35%.

### WinoBias (lower = less gender-coded)

| Model | Family | Release | Bias Score |
|---|---|---|---|
| smollm2-360m | huggingface | 2024-11 | **7.2** |
| qwen25-05b | alibaba | 2024-09 | **16.9** |

*Bias score = pro-accuracy − anti-accuracy. 0 = no gendered bias in pronoun resolution.*

### CV Screening

smollm2-135m: mean score **82.5/100** (std 3.54, 4,800 scored CVs). Full statistical analysis (group means, 95% CI, Cohen's d, variance breakdown by gender, ethnicity, university prestige) writes to `results/analysis_summary.txt`.

### Cross-model heatmap

<img src="figs/cross_model_heatmap.png" width="700" alt="Cross-model benchmark heatmap">

Every model × benchmark combination, sorted by mean bias (lower = greener).

### Model size vs bias

<img src="figs/size_vs_bias.png" width="700" alt="Model size vs bias">

Parameter count vs bias score per benchmark, with trend lines.

### Per-family comparison

<img src="figs/family_comparison.png" width="700" alt="Per-family bias comparison">

Mean bias score grouped by model family.

### StereoSet categories

<img src="figs/stereoset_categories.png" width="700" alt="StereoSet category breakdown">

Stereotype scores by category (gender, race, religion, profession) per model.

### WinoBias pronoun accuracy

<img src="figs/winobias_pronouns.png" width="700" alt="WinoBias pronoun accuracy">

Accuracy per pronoun (he/she/they) across models. Large gaps indicate gendered prediction bias.

### Demographic group output length

<img src="figs/demographic_groups.png" width="700" alt="Demographic group output length">

Normalised output length by demographic group per model. Disparities suggest demographic bias in generation length.

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

# Run all benchmarks on one model
uv run python -m slm_bias_testing.runner smollm2-135m --benchmark all

# Run a specific benchmark with limited samples
uv run python -m slm_bias_testing.runner smollm2-135m --benchmark stereoset --max-samples 20

# Batch: multiple models, one benchmark
uv run python scripts/run_experiments.py \
  --models smollm2-135m,smollm2-360m \
  --benchmarks stereoset \
  --max-samples 20

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
  temporal_trends.png         — Bias score vs release date (committed, shown above)
  family_comparison.png       — Per-family bias comparison
  cross_model_heatmap.png     — Model × benchmark heatmap
  size_vs_bias.png            — Params vs bias scatter
  stereoset_categories.png    — StereoSet per-category breakdown
  winobias_pronouns.png       — WinoBias accuracy per pronoun
  demographic_groups.png      — Output length by demographic group
```

---

## Project structure

```
src/slm_bias_testing/
  registry.py       — Model definitions (name → ollama tag)
  runner.py         — CLI entry point for running benchmarks
  benchmark.py      — CV screening benchmark
  call_api.py       — Ollama model API client
  temporal.py       — Temporal analysis & trend plots
  analysis.py       — Statistical helpers (CI, Cohen's d, variance)
  benchmarks/
    stereoset.py         — StereoSet benchmark
    winobias.py          — WinoBias gender coreference benchmark
    demographic_bias.py  — Output length disparity benchmark
  data/
    cvs.py               — 600-CV factorial corpus (gender × ethnicity × prestige × quality × template)
    cv_template.py       — CV text templates
    job_description.py   — Fixed Junior Data Analyst JD

scripts/
  run_experiments.py  — Batch runner (kill-safe, skips completed)

tests/                — 114 tests
```

---

## Prerequisites

- [Ollama](https://ollama.ai) — all models run locally
- `uv` (or `pip`) for Python dependencies

---

## Tests

```bash
uv run pytest tests/ -q
uv run ruff check src tests
```

---

*Questions? Open an issue or ping @William-Dennis.*
