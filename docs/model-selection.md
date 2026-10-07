# Model Selection

Every model in the registry (`src/slm_bias_testing/registry.py`) is chosen
against an external reference rather than by gut feel. This document records
the reference, what it does and does not cover, how the current registry maps
to it, and the shortlist of candidates proposed by issue #41.

## Source: Open SLM Leaderboard

**Reference:** <https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard>
**Retrieved:** 2026-10-07

Benchmark suite per model: ARC, HellaSwag, PIQA, ARC-Challenge, ArithMark-3,
ArithMark-2, plus two derived ranks:

- **Avg** — weighted mean of HellaSwag, combined ARC, PIQA, ArithMark-3
- **Intelligence Index (II)** — same inputs, each chance-normalized
  (HellaSwag/ARC ×1.0, PIQA ×1.0, ArithMark-3 ×0.65; chance = 25/25/50/25).
  0 = random guessing, 100 = perfect. **Ranks below use II.**

### Retrieving the data

The space is a static site: one `index.html` with the data embedded as a JS
array — no API, no JSON endpoint. To refresh this document:

```bash
curl -s "https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard/raw/main/index.html" -o /tmp/lb.html
# the model table is the `const MODELS = [ ... ]` array (~line 1870);
# each entry: name, org, params, paramsDisplay, arc, hellaswag, piqa,
#             arcChall, arithmark3, arithmark2, links
```

### Coverage caveat

The leaderboard holds **207 models, all between 59k and 157M parameters**.
It grounds the *tiny* end of the SLM spectrum only. Anything above ~160M
(including most of our registry) cannot be ranked from it and must be
grounded from the model author's own cards/papers instead.

## Current registry against the source

| registry entry | params | leaderboard standing |
|---|---|---|
| smollm-135m | 135M | **#4 / 207** — II 25.7 (ARC 56.3, HS 42.7, PIQA 68.3) |
| smollm2-135m | 135M | **#2 / 207** — II 27.1 (ARC 58.6, HS 43.2, PIQA 68.4) |
| smollm-360m | 360M | out of coverage (>157M) |
| smollm2-360m | 360M | out of coverage |
| qwen25-05b | 500M | out of coverage |
| qwen25-15b | 1.5B | out of coverage |
| qwen35-08b | 800M | out of coverage |
| qwen3-06b | 600M | out of coverage |
| gemma3-270m | 270M | out of coverage |
| gemma3-1b | 1.0B | out of coverage |
| granite4-350m | 350M | out of coverage |
| lfm2-350m | 350M | out of coverage |
| lfm2-700m | 700M | out of coverage |
| llama32-1b | 1.0B | out of coverage |
| tinyllama | 1.1B | out of coverage |
| stablelm2-16b | 1.6B | out of coverage |

Takeaway: our two leaderboard-covered picks (the 135M SmolLM pair) sit at
#2 and #4 of 207 — the tiny-end entries are well grounded. The other 14 are
outside this source's range; they were selected on family diversity and
Ollama availability (see `registry.py` metadata) and should be re-checked
against their own model cards when accuracy claims matter.

## Proposed registry additions

Criteria: II evidence from the leaderboard, size inside our SLM band
(≤ ~160M for leaderboard-grounded picks), and a workable path to running it
locally. Candidates (top-II models absent from the registry, 2026-10-07):

| candidate | org | params | II | Ollama |
|---|---|---|---|---|
| cagliostro-v3.5 | benchlabs | 146M | 27.5 | not published |
| cagliostro-v3 | benchlabs | 146M | 26.5 | not published |
| GPT-X2.5-135M | axiomiclabs | 135M | 25.2 | not published |
| Haidass1.5-143M | dalab | 143M | 25.1 | not published |
| BananaMind-2-Pro | bananamind | 139M | 25.0 | not published |
| MobileLLM-R1-140M-base | facebook | 140M | 24.6 | not published |
| Speck2-140M-Instruct | specklabs | 141M | 21.3 | not published |

(`GPT-X3-Preview-1`, II 25.7, is skipped as a preview release;
availability checked against the Ollama library search on 2026-10-07 —
none of the above are published as Ollama models.)

### Why none can be added yet, and the paths

The registry runs exclusively on Ollama tags, and none of these candidates
is in the Ollama library. Three routes, in order of effort:

1. **Convert and publish locally** — fetch the HF weights, convert to GGUF,
   `ollama create` a local tag. Works today, no code changes; the tag only
   exists on machines that built it (document the recipe in the PR).
2. **Add a second backend to the runner** — a registry `api` field
   (`"chat" | "systemone" | "hf"`) so entries can point at HuggingFace or
   at Ollama's typed-decision endpoint. Issue #46 already requires the
   `api` field for decision models; the HF backend is an extension of the
   same work.
3. **Wait for upstream** — request publication in the Ollama library and
   re-check (the space's models are niche; don't count on it).

Recommendation: take route 1 for one or two candidates as a pilot (cheapest,
validates the adds are worth benchmarking), and fold route 2 into the #46
registry work if we want these to be first-class citizens.

Decision-model entries (`laya`, `tev1`, …) are a separate, in-progress
addition under issue #46 — tracked there until their own methodology doc
lands.

## Limitations

- The leaderboard measures academic task accuracy (ARC/HellaSwag/PIQA/
  ArithMark). It grounds *which models are worth testing for bias* — it says
  nothing about their bias behaviour. Bias numbers only come from running
  our own benchmarks.
- II values are recomputed from the embedded data on the retrieval date
  above; the space updates continuously, so ranks drift. Re-retrieve before
  extending this document.
