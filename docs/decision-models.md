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

> **Status: implemented** (issue #46, see
> [Integration](#integration-decision-models-as-benchmark-models) below).
> Deviations from this plan: the client lives in `decision_models.py`
> (next to the transport), and records keep the standard `score` column
> (rounded `score_continuous`; the raw typed payload stays in `response`)
> instead of a separate `score_source` field.

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

## Instrument v2: 500-token budget, step-10 scoring (scored sweep)

A second, **deliberately separate** instrument lives in
`src/slm_bias_testing/decision_instrument.py` — it does not touch or reuse
the generative prompt builders (`build_base_prompt`/`cv_prompt`), the full
JD, or the CV templates; only the CV corpus is shared.

| piece | v2 choice |
|---|---|
| state | `Screening for: {JD_STUB}` + byte-identical CV — generative XX/100 boilerplate dropped |
| JD | fresh ~30-token requirements stub (`JD_STUB`); the 295-word JD does not fit any 500-token budget (it alone costs 425–461 tokens) |
| question | one `score` question, 11 levels `0,10,...,100` (laya allows 2–26), terse instructions; `advance` noul dropped |
| budget | `BUDGET_TOKENS = 500`, recorded per record as `budget_ok` |
| outputs | `score_discrete` (argmax x 10), `score_continuous` (probability-weighted x 10), all 11 probabilities, `confidence` |

Run: `uv run python scripts/benchmark_decision_model.py --score-sweep
--json docs/data/decision-sweep-2026-10-07.json` — 600 CVs x 3 tags,
**12–20 s per tag** (model loaded once per tag).

### Fit: exactly 480/600 on every tag, for a structural reason

State length is **bimodal by CV template** (English/typed tokenizer):
templates a–d use 355–438 tokens, template_e uses 518–526.

| tag | scored | ctx errors | <=500 budget | max state | wall |
|---|---|---|---|---|---|
| `laya` (512 ctx) | **480/600** | 120 (all template_e) | 480 | 526 | 15.8 s |
| `322m-multilingual` (1024) | **600/600** | 0 | 480 | 586 (mmBERT counts ~15% higher) | 12.3 s |
| `421m-typed-decisions` (1024) | **600/600** | 0 | 480 | 526 | 20.3 s |

- The 500-token budget holds for **4 of 5 templates on every tag**;
  template_e's long layout costs 518–586 tokens and cannot fit 500 with a
  byte-identical CV on any tokenizer. Strictly-≤500 for all 600 is not
  achievable without truncating CVs (rejected: content is the measured
  variable).
- English's 512 window loses template_e regardless of stub size — even the
  longest CV alone (~454 tokens) exceeds its 439-token limit for this
  question. template_b sits **1 token** under that limit (max 438).
- mmBERT (multilingual) tokenizes the same text ~8–15% higher; its
  template_e states reach 586, still far below its ~890 limit.

### Scores: continuous carries the signal, argmax is noise

| tag | score_continuous mean +/- sd (range) | confidence (mean) | top-bin prob (mean; uniform = 0.091) |
|---|---|---|---|
| `laya` (n=480) | **52.5 +/- 3.9** (44.7-58.3) | 0.034 | 0.162 |
| `322m-multilingual` (n=600) | **40.3 +/- 8.0** (21.2-47.3) | 0.110 | 0.266 |
| `421m-typed-decisions` (n=600) | **56.7 +/- 4.5** (47.1-60.8) | 0.038 | 0.164 |

Three findings that shape everything downstream:

1. **The probability distributions are near-uniform** (top bin only
   0.16 vs 0.091 uniform; `confidence` 0.03-0.11). CV screening is far
   from laya's email/routing training distribution — the model is telling
   us it is unsure. (`confidence` here is normalised entropy over the level
   probabilities - a dispersion measure, **not** evidence that the
   probabilities are calibrated; shipped checkpoints are documented as
   unvalidated for calibration.)
   Consequently **`score_discrete` (argmax) is effectively noise**: e.g.
   `laya` assigns 335/480 CVs to level 100 while its weighted mean is 52.
   **Headline number = `score_continuous`**; `score_discrete` stays in the
   records for provenance but must not be reported as "the score".
2. **Checkpoints offset each other but agree on ordering**: pairwise
   Pearson r on `score_continuous` = **0.90-0.96** across all three tags,
   while means differ by up to **16 points** (multilingual runs ~14-16
   lower than typed-decisions). Pin one tag per comparison; never mix.
3. **First-look factor separation (typed-decisions):** demographic factors
   show **no visible effect** (gender 56.6-56.7, ethnicity 56.4-56.7,
   prestige 56.6-56.7, A-levels 56.6-56.7 — all deltas <= 0.3 points),
   while **template dominates** (46.3-60.0 across tags, e.g. template_d
   ~8-12 points below the rest). The instrument clearly reads content;
   whether demographic insensitivity is real or a sensitivity limit of a
   near-uniform head is the first question for the follow-up analysis.

### Limitations of v2 (honest list)

- Out-of-domain task for the checkpoint family (triage/routing -> CVs);
  near-uniform probabilities are the visible symptom.
- Numeric level labels (`"0"..."100"`) may contribute to the flat
  distributions; descriptive criteria (e.g. reject..top-candidate) would
  trade away literal step-10 labels — worth a follow-up A/B.
- English tag is partial (480/600, template_e only); budget strict on
  480/600.
- The state uses a stub JD, so v2 numbers are **not comparable** to the
  generative instrument's full-JD scores either — different instrument,
  recorded as `instrument: decision-v2` in the sweep JSON.

## Integration: decision models as benchmark models

`api: "systemone"` models run the CV-screening benchmark through the same
records/analysis pipeline as chat models (issue #46). Four registry
entries — `laya-english`, `laya-multilingual`, `laya-typed-decisions`
(Convai, encoder-only) and `nimble` (Bespoke, 9B decoder-only, the one
deliberate >1B exception to the small-model framing, issue #53) —
carry the required `ModelMeta.api` field (`"chat"` for everything
else).

| piece | choice |
|---|---|
| client | `SystemOneClient` (`decision_models.py`) — `Predictor`-compatible: `predict(prompt, temperature)` posts the prompt **as state** with the fixed `score_question()`, returns `<raw answers JSON>\nNN/100` so `parse_score()` and the `response` column work unchanged |
| score mapping | record `score` = **round-half-up of `score_continuous`** (the v2 headline), clamped 0–100 — argmax `score_discrete` is constant-ish noise and would show zero variance; the raw payload in `response` keeps the full distribution |
| state | `decision_base_frame()` + the existing `cv_prompt()` = `decision_state()` byte-for-byte (all 600 corpus CVs verified); no second prompt builder |
| determinism | `n_runs` is **forced to 1** (explicit `--n-runs > 1` is rejected before the run) — repeated runs add nothing, per the spike's byte-identical finding |
| pool | skipped: the Node.js pool speaks the chat API only; sequential `SystemOneClient` calls take 12–20 s per full corpus anyway |
| scope | CV screening only — decision models cannot generate text, so stereoset/winobias/demographic-bias are skipped with a logged error |
| provenance | `cv-screening.json` records `"api": "systemone"` and `temperature: null` (ignored by the protocol) |

### Committed results (`results/{model}/cv-screening/`)

Full 600-CV corpus, `n_runs=1`, rerun with
`uv run python scripts/run_benchmarks.py --models laya-english,laya-multilingual,laya-typed-decisions,nimble --benchmark cv-screening`:

| model | scored | mean (int score) | std | attrition |
|---|---|---|---|---|
| `laya-english` | 480/600 | 52.5 | 3.9 | 120 ctx errors (all template_e, 512-token window) |
| `laya-multilingual` | 600/600 | 40.3 | 8.0 | — |
| `laya-typed-decisions` | 600/600 | 56.7 | 4.5 | — |
| `nimble` (9B) | 600/600 | 65.1 | 14.6 | — (8192-token window) |

Means match the v2 sweep's `score_continuous` means to 0.1 for the laya
tags — the benchmark path is the same instrument. Each model's
`analysis_summary.txt` has the group CIs, Holm-corrected pairwise tests
and variance breakdown.

### Four-model bias comparison: protected vs merit factors

The benchmark's factor set mixes two kinds of signal: **protected
proxies** (`name_gender`, `name_ethnicity`) and **merit factors**
(`university_prestige`, `a_level_quality` — a recruiter *should* weigh
these). Splitting them changes the reading:

| model | protected: max gap | max \|d\| | sig pairs | merit: max gap | max \|d\| | sig pairs |
|---|---|---|---|---|---|---|
| `laya-english` | 0.65 | 0.168 | 0 | 0.54 | 0.139 | 0 |
| `laya-multilingual` | 0.81 | 0.099 | 0 | 0.40 | 0.050 | 0 |
| `laya-typed-decisions` | 0.29 | 0.066 | 0 | 0.16 | 0.034 | 0 |
| `nimble` (9B) | **1.17** | 0.080 | 0 | **11.91** | **0.897** | **2** |

(sig = Holm-corrected p < 0.05 over all pairwise comparisons in the
factor.)

Findings:

1. **No model shows significant protected-attribute bias** — zero
   Holm-significant gender/ethnicity pairs anywhere; worst protected
   effect is trivial (|d| = 0.17, laya-english). nimble's largest
   protected gap (1.17, ethnicity) is non-significant and points
   *black-african highest, white-british lowest* — no anti-protected
   direction.
2. **nimble is the first decision model that reads merit**: a-level
   quality separates its scores (high 69.7 vs low 57.8, d = 0.90,
   p_holm ≈ 0; university prestige +2.9), and its std (14.6 vs laya's
   3.9–8.0) shows real cross-CV spread. `a_level_quality` explains
   12.8% of its variance (laya: ≤ 0.04%).
3. **laya's flatness is instrument sensitivity, not proven fairness** —
   its near-uniform heads ignore *everything* content-shaped
   (merit gaps ≤ 0.54, n.s.; template still dominates), so "no bias"
   there remains the floor-effect caveat from the v2 sweep. nimble's
   pattern — strong merit sensitivity, protected deltas ≤ 1.17 — is
   what an unbiased-but-attentive screener looks like.

Caveat: operating points differ (means 40–65), and laya-english scores
480/600 — compare structures, not absolute scores.

## Reproducing the spike

```bash
# Ollama >= 0.40 with the four tags pulled
ollama pull laya && ollama pull laya:322m-multilingual-mlx-fp16
ollama pull laya:421m-typed-decisions-mlx-fp16
uv run python scripts/benchmark_decision_model.py --n 50 --threads 4 --json /tmp/spike.json
# scored 600-CV sweep with the v2 instrument (see section above):
uv run python scripts/benchmark_decision_model.py --score-sweep --json /tmp/sweep.json
```

Model API references: Ollama blog (2026-09-29) *Ollama now supports
Jev-style decision models*; <https://ollama.com/library/laya>;
<https://github.com/NandhaKishorM/laya>.
