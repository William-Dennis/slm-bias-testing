# Decision Models — Local Spike (issue #46)

Non-autoregressive "System 1" decision models answer **named, typed
questions** over a text state in one encoder pass — `noul` (P(yes)),
`choice` (label + distribution), `score` (ordinal levels) — instead of
generating text. Nothing to parse, nothing to hallucinate, no `NN/100` regex.

This document is the spike requested by issue #46: **how quickly can we run
this locally, and what output shapes match our benchmarks?** Raw numbers:
[`data/decision-spike-2026-10-07.json`](data/decision-spike-2026-10-07.json)
(produced by `scripts/benchmark_decision_model.py`).

## The ecosystem in one paragraph

TypeSafe's **Jev** defines the API contract (`/v1/systemone`: state +
questions → typed answers with probabilities/confidence). Since Ollama 0.35
the contract is served **natively and locally** (Ollama blog, 2026-09-29) —
no API key, no SDK (plain HTTP POST). The hosted Jev model itself still
requires a key and is **out of scope** by decision. Ollama also ships
`nimble` (9B, Bespoke) and `tev1` (4B/0.8B, Together); this spike is
**laya-only** (decision), a 421M ModernBERT / 322M mmBERT model family from
Convai Innovations.

## Spike setup

- **Machine:** Apple M5 Pro (arm64), macOS.
- **Ollama:** upgraded 0.34.0 → **0.40.0** (laya's floor; `/v1/systemone`
  needs ≥ 0.35). Daemon headless (`ollama serve`), CLI at `~/.local/bin`.
- **Tags pulled (all four published):**

| tag | digest | size | context |
|---|---|---|---|
| `laya` (= `laya:421m-english-mlx-fp16`, same digest `aaba5574a1e1`) | aaba5574a1e1 | 846 MB | 512 |
| `laya:322m-multilingual-mlx-fp16` | 63b6e37ecedb | 678 MB | 1024 |
| `laya:421m-typed-decisions-mlx-fp16` | c62bc110244c | 846 MB | 1024 |

  On macOS every published laya tag is **MLX-backed — the default tag *is*
  the English MLX build** (identical digests), so there is no non-MLX
  baseline to compare against on this platform.
- **Protocol per model:** unload (`ollama stop`) → 1 cold request → 50 warm
  requests on a short control state → 50 warm requests on the **real CV
  screening prompt** (`build_base_prompt` + `cv_prompt`, corpus CV #1:
  4,116 chars / 553 words / 847 tokens) with both decision questions in one
  payload → 50 requests across 4 threads → unload.
- **Determinism:** answers are byte-identical across every run
  (multilingual `noul = 0.8278846144565452` in the smoke run, the full run
  and isolated probes). There is no sampling — **repeated runs add nothing**;
  for decision models `n_runs > 1` must not be used, and variation comes
  only from the 600-CV cross-section. (The generative protocol's
  `temperature=1.0` × 10 runs does not apply; see
  [cv-screening-methodology.md](cv-screening-methodology.md).)

## Latency and throughput (n=50 per phase)

| model | ctx | cold load | short p50/p95 | CV prompt p50/p95 | throughput rps @4 threads (short) | CV prompt |
|---|---|---|---|---|---|---|
| `laya` (english) | 512 | 1517 ms | 13 / 13 ms | **rejected** | 101.7 | error, 0/50 |
| `laya:322m-multilingual-mlx-fp16` | 1024 | 735 ms | 8 / 11 ms | 48 / 49 ms | 197.3 | 50/50 ok |
| `laya:421m-typed-decisions-mlx-fp16` | 1024 | 329 ms | 13 / 14 ms | 101 / 103 ms | 98.8 | 50/50 ok |

Reading:

- **Cold = first-request cost** (model load + encode), 0.3–1.5 s across
  runs (volatile; once per load). Warm single-question decisions are
  **8–13 ms**; the full CV prompt (~900-token state, 2 questions) is
  **48–103 ms**.
- Throughput at 4 threads on the short state: **99–197 successful
  decisions/s** (`rps_ok`; attempted rate reported separately).
  At realistic CV-prompt sizes (full 600-CV sweep, sequential): multilingual
  **24/s**, typed-decisions **9.4/s**.
- `input_tokens` in the responses **sums one encoder row per question**
  (the state is re-encoded per question: ~1,760–1,900 for 2 questions on a
  ~900-token state) — an accounting figure, not per-pass cost.
- The english tag fails the CV prompt with an explicit error, never silent
  truncation: `state has 847 tokens; limit is 477 with this question`.

### Full-corpus fit (all 600 CV prompts, 2 questions each)

| tag | fits | rejected | throughput |
|---|---|---|---|
| `laya` (512 ctx) | 0 / 600 | 600 | 2 s total |
| `322m-multilingual` (1024) | **480 / 600** | 120 (state > 989 tok) | 24 s total |
| `421m-typed-decisions` (1024) | **600 / 600** | 0 | 64 s total |

Prompt sizes span 3,927–4,650 chars (median 4,124). **`421m-typed-decisions`
is the only tag that runs the entire corpus unmodified**, and a complete
600-CV pass costs **~64 seconds** of model time — versus hours for the
generative path (6,000 sampled attempts).

### Answers disagree across checkpoints

Same prompt, same questions, deterministic:

| tag | `advance` (noul) | `strength` (score, 3 levels) | confidence |
|---|---|---|---|
| `322m-multilingual` | **0.83** | 1.52 | 0.32 |
| `421m-typed-decisions` | **0.64** | 1.47 | 0.23 |

Checkpoint choice moves the headline number more than any prompt detail —
the two must not be averaged or compared silently. `score` is a continuous
value in **[0, k−1]** for k levels (here 0–2), not [0,1]; `noul` carries no
`confidence` field in the payload.

## Output-shape compatibility with our benchmarks

| benchmark | requires today | fit | mapping if integrated |
|---|---|---|---|
| CV screening | `NN/100` text from a sampling prompt | **partial** | `noul` P(advance)×100 (0–100, calibration-dependent) **or** `score` with k levels (coarse ordinal). Either way the number is no longer "LLM score out of 100" — provenance must record `score_source: noul_probability \| score_ordinal`. Deterministic → `n_runs=1`. |
| StereoSet | 0–100 appropriateness per continuation | **partial** | `score` over ordered levels ("inappropriate … appropriate"), or `noul` "is this continuation appropriate?" — changes the scale from the published one. |
| WinoBias | free-text answer containing the antecedent | **rework** | natural fit as `choice` over candidate entities — but the prompt must be redesigned as a typed coreference question; results not comparable with existing generative runs. |
| Demographic bias (output length) | generated text length | **incompatible** | no generation exists. Nothing to map — by construction out of scope. |

Context is the gating constraint: decision models see **512/1024 tokens**,
not the chat models' 4k–32k. Only `421m-typed-decisions` clears the current
corpus (600/600); any corpus or prompt growth must re-run the fit sweep
(`scripts/benchmark_decision_model.py --sweep` measures exactly this).

## What an integration (PR I) would add

1. **Registry:** `ModelMeta` gains `api: "chat" | "systemone"`; entries for
   the chosen tag(s) (recommend `laya:421m-typed-decisions-mlx-fp16` as the
   only corpus-complete one; keep `322m-multilingual` as the faster
   partial-corpus comparison — they disagree 0.83 vs 0.64, which is itself a
   finding worth reporting).
2. **Client:** `SystemOneClient` in `model_clients.py` — HTTP POST
   `/v1/systemone`, no Node pool (the pool speaks `/api/chat`); runner
   skips pool lifecycle for `api="systemone"` entries.
3. **Adapters:** per-benchmark question builders + answer→record mapping
   with `score_source` provenance; `n_runs` forced to 1 (deterministic);
   attrition counts token-overflow rejections.
4. **Runs:** full 600-CV pass (~1 min model time) then commit `results/`
   + regenerated figures, retiring the README stale-numbers warning.

**Recommendation:** proceed with PR I. The typed path is 1–2 orders of
magnitude faster than the generative path, needs no prompt parsing, fails
loud on context overflow, and the determinism removes a whole class of
variance — at the honest cost of a different score semantics that must stay
visible in provenance.

## Reproducing the spike

```bash
# Ollama >= 0.40 with the four tags pulled
ollama pull laya && ollama pull laya:322m-multilingual-mlx-fp16
ollama pull laya:421m-typed-decisions-mlx-fp16
uv run python scripts/benchmark_decision_model.py --n 50 --threads 4 --json /tmp/spike.json
```

Model API references: Ollama blog (2026-09-29) *Ollama now supports
Jev-style decision models*; <https://ollama.com/library/laya>;
<https://github.com/NandhaKishorM/laya>.
