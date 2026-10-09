/* Synthetic result generation. Latent per-model profiles drive every
   number; intervals and effects are computed from generated counts with
   the estimators in stats.ts — never hand-typed. */
import type { GroupStat, ModelBenchmarkResult, PairwiseStat } from "./types";
import { BENCHMARKS, benchmarkById } from "./benchmarks";
import { MODELS } from "./models";
import { cohensH, holm, newcombe, r2, rand, rng, twoPropP, wilson } from "./stats";

/** Latent disparity profile per model (percentage-point scale unless noted).
 *  [stereo, wino, cvPts, demoPct, refusal, tox, help, dialect, lend, accomPts, relig, visa, accuracy, latencyMs, costCents] */
const PROFILES: Record<string, number[]> = {
  "smollm-135m": [68, 16.5, 4.8, 38, 9.5, 6.2, 11, 14, 7.5, 1.8, 5.4, 12, 52, 410, 0.4],
  "smollm-360m": [63, 7.0, 3.9, 22, 7.0, 4.8, 8, 11, 6.0, 1.4, 4.2, 9, 55, 480, 0.5],
  "smollm2-135m": [55, 3.5, 2.1, 12, 4.5, 2.6, 5, 7, 3.2, 0.8, 2.4, 6, 61, 390, 0.4],
  "smollm2-360m": [58, 7.2, 2.8, 18, 5.5, 3.4, 6, 8, 4.1, 1.0, 3.0, 7, 63, 520, 0.6],
  "qwen25-05b": [66, 16.9, 4.2, 31, 8.0, 5.5, 9, 12, 6.8, 1.6, 4.8, 10, 58, 450, 0.5],
  "qwen25-15b": [60, 11.0, 3.1, 24, 6.2, 4.0, 7, 9, 5.0, 1.2, 3.6, 8, 66, 900, 1.1],
  "qwen35-08b": [57, 8.5, 2.6, 19, 5.0, 3.2, 6, 8, 4.2, 1.0, 3.0, 7, 68, 700, 0.9],
  "qwen3-06b": [56, 6.0, 2.2, 15, 4.2, 2.8, 5, 6, 3.4, 0.8, 2.5, 6, 64, 620, 0.7],
  "gemma3-270m": [53, 4.2, 1.6, 9, 3.2, 2.0, 4, 5, 2.4, 0.6, 1.8, 5, 62, 430, 0.5],
  "granite4-350m": [54, 5.1, 1.9, 11, 3.8, 2.4, 4.5, 6, 2.8, 0.7, 2.1, 5, 65, 560, 0.7],
  "lfm2-350m": [59, 9.2, 3.0, 21, 6.0, 3.8, 7, 9, 4.6, 1.1, 3.3, 8, 60, 500, 0.6],
  "lfm2-700m": [57, 7.8, 2.5, 17, 5.2, 3.1, 6, 8, 3.9, 0.9, 2.9, 7, 64, 680, 0.8],
  "llama32-1b": [61, 10.5, 3.4, 26, 6.8, 4.4, 7.5, 10, 5.4, 1.3, 3.9, 9, 67, 950, 1.2],
  tinyllama: [64, 12.0, 3.8, 29, 7.4, 4.9, 8.5, 11, 6.0, 1.5, 4.3, 9, 54, 800, 1.0],
  "stablelm2-16b": [62, 11.4, 3.5, 27, 7.0, 4.5, 8, 10, 5.6, 1.4, 4.0, 9, 66, 1100, 1.4],
  "gemma3-1b": [54, 5.5, 1.8, 13, 3.6, 2.2, 4.5, 6, 2.7, 0.7, 2.0, 5, 69, 980, 1.2],
  "laya-english": [52, 2.8, 1.2, 8, 2.6, 1.8, 3.5, 4, 2.0, 0.5, 1.5, 4, 58, 300, 0.3],
  "laya-multilingual": [51, 2.2, 1.0, 7, 2.2, 1.5, 3, 3.5, 1.7, 0.4, 1.3, 3.5, 57, 260, 0.3],
  "laya-typed-decisions": [50.5, 1.8, 0.8, 6, 1.8, 1.2, 2.5, 3, 1.4, 0.3, 1.0, 3, 59, 240, 0.3],
  nimble: [52, 3.0, 1.1, 8, 2.4, 1.6, 3, 4, 1.8, 0.5, 1.4, 4, 78, 2400, 3.8],
};

const prof = (modelId: string, i: number): number =>
  (PROFILES[modelId] ?? PROFILES["smollm2-135m"])[i];

export const modelLatency = (id: string): number => prof(id, 13);
export const modelCostC = (id: string): number => prof(id, 14);
export const modelAccuracy = (id: string): number => prof(id, 12);

interface GroupSpec {
  groups: string[];
  nPerGroup: number[];
  baseRate: number; // baseline proportion
  gapIdx: number; // index into PROFILES row
  gapBetween: [number, number]; // group indices the gap separates
}

/** Group layouts per benchmark. Rates are proportions of a binary outcome. */
function specFor(benchmarkId: string): GroupSpec {
  const b = benchmarkById(benchmarkId);
  const t = b.tasks;
  switch (benchmarkId) {
    case "stereoset":
      return { groups: ["gender", "race", "religion", "profession"], nPerGroup: [t / 4, t / 4, t / 4, t / 4], baseRate: 0.55, gapIdx: 0, gapBetween: [0, 3] };
    case "winobias":
      return { groups: ["type1-pro", "type1-anti", "type2-pro", "type2-anti"], nPerGroup: [t / 4, t / 4, t / 4, t / 4], baseRate: 0.62, gapIdx: 1, gapBetween: [0, 1] };
    case "refusal-disparity":
      return { groups: ["standard", "AAVE", "regional-UK", "ESL"], nPerGroup: [t / 4, t / 4, t / 4, t / 4], baseRate: 0.08, gapIdx: 4, gapBetween: [0, 1] };
    case "toxicity-gap":
      return { groups: ["race-mention", "religion-mention", "gender-mention", "neutral"], nPerGroup: [t / 4, t / 4, t / 4, t / 4], baseRate: 0.05, gapIdx: 5, gapBetween: [0, 3] };
    case "helpfulness-pairwise":
      return { groups: ["older-framing", "younger-framing", "access-needs", "neutral"], nPerGroup: [t / 4, t / 4, t / 4, t / 4], baseRate: 0.55, gapIdx: 6, gapBetween: [0, 3] };
    case "dialect-robustness":
      return { groups: ["standard", "AAVE", "IndE", "SgE"], nPerGroup: [t / 4, t / 4, t / 4, t / 4], baseRate: 0.7, gapIdx: 7, gapBetween: [0, 1] };
    case "lending-fairness":
      return { groups: ["male-cue", "female-cue", "older-cue", "younger-cue"], nPerGroup: [t / 4, t / 4, t / 4, t / 4], baseRate: 0.6, gapIdx: 8, gapBetween: [0, 1] };
    case "religion-respect":
      return { groups: ["christian", "muslim", "jewish", "hindu"], nPerGroup: [t / 4, t / 4, t / 4, t / 4], baseRate: 0.04, gapIdx: 10, gapBetween: [0, 1] };
    case "visa-info":
      return { groups: ["UK", "India", "Nigeria", "Brazil"], nPerGroup: [t / 4, t / 4, t / 4, t / 4], baseRate: 0.68, gapIdx: 11, gapBetween: [0, 2] };
    default:
      return { groups: ["a", "b"], nPerGroup: [t / 2, t / 2], baseRate: 0.5, gapIdx: 1, gapBetween: [0, 1] };
  }
}
function sha12(s: string): string {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < s.length; i++) {
    const ch = s.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  const h = (h1 ^ h2) >>> 0;
  return h.toString(16).padStart(8, "0") + "abcd";
}

/** Build a rate-based result: group proportions with Wilson CIs,
 *  Newcombe gap interval, Cohen's h, Holm-adjusted pairwise p-values. */
function buildRateResult(
  modelId: string,
  benchmarkId: string,
  label: string,
  unit: "pp" | "%"
): ModelBenchmarkResult {
  const b = benchmarkById(benchmarkId);
  const spec = specFor(benchmarkId);
  const r = rng(`${modelId}|${benchmarkId}|rates`);
  const gapPp = prof(modelId, spec.gapIdx);
  const n = spec.nPerGroup.map((x) => Math.max(30, Math.round(x)));
  const rates = spec.groups.map((_, gi) => {
    let rate = spec.baseRate + rand(r, -0.02, 0.02);
    if (gi === spec.gapBetween[0]) rate += gapPp / 200;
    if (gi === spec.gapBetween[1]) rate -= gapPp / 200;
    return Math.min(0.97, Math.max(0.01, rate));
  });
  const ks = rates.map((p, gi) => Math.round(p * n[gi]));

  const groups: GroupStat[] = spec.groups.map((g, gi) => {
    const [lo, hi] = wilson(ks[gi], n[gi]);
    const sd = Math.sqrt(rates[gi] * (1 - rates[gi]));
    return { group: g, n: n[gi], mean: r2(rates[gi] * 100), sd: r2(sd * 100), ciLow: r2(lo), ciHigh: r2(hi) };
  });

  // All pairwise comparisons with Holm correction
  const rawP: number[] = [];
  const pairs: { i: number; j: number; d: number; lo: number; hi: number }[] = [];
  for (let i = 0; i < spec.groups.length; i++) {
    for (let j = i + 1; j < spec.groups.length; j++) {
      const [lo, hi] = newcombe(ks[i], n[i], ks[j], n[j]);
      pairs.push({ i, j, d: cohensH(rates[i], rates[j]), lo, hi });
      rawP.push(twoPropP(ks[i], n[i], ks[j], n[j]));
    }
  }
  const adj = holm(rawP);
  const pairwise: PairwiseStat[] = pairs.map((p, k) => ({
    g1: spec.groups[p.i],
    g2: spec.groups[p.j],
    effect: r2(p.d),
    effectKind: "Cohen's h",
    ciLow: r2(p.lo),
    ciHigh: r2(p.hi),
    pHolm: adj[k] === null ? null : r2(adj[k] as number),
    significant: adj[k] !== null && (adj[k] as number) < 0.05,
  }));

  // Headline = largest absolute pairwise gap
  const order = [...pairwise].sort(
    (a, b) => Math.abs(b.ciLow + b.ciHigh) - Math.abs(a.ciLow + a.ciHigh)
  );
  const top = order[0];
  const mid = (top.ciLow + top.ciHigh) / 2;
  const totalN = n.reduce((a, x) => a + x, 0);
  const sig = totalN < 200 ? "uncertain" : top.significant ? "significant" : "not-significant";

  const model = MODELS.find((m) => m.id === modelId) ?? MODELS[0];
  const nRuns = model.api === "systemone" ? 1 : 10;
  const stamp = `2026-09-${String(10 + (ks[0] % 18)).padStart(2, "0")}T14:20:00Z`;
  const history = [0, 1, 2, 3].map((k) => ({
    runId: `run-${202606 + k * 1}`,
    timestamp: `2026-0${6 + k}-12T10:00:00Z`,
    value: r2(Math.abs(mid) + rand(r, -0.9, 0.9)),
    n: totalN,
  }));

  return {
    modelId,
    benchmarkId,
    headline: {
      label, value: r2(Math.abs(mid)), unit,
      ciLow: r2(Math.min(Math.abs(top.ciLow), Math.abs(top.ciHigh))),
      ciHigh: r2(Math.max(Math.abs(top.ciLow), Math.abs(top.ciHigh))),
      n: totalN, effect: top.effect, effectKind: top.effectKind,
      pHolm: top.pHolm, sig,
      note: `Largest observed pairwise gap (${top.g1} vs ${top.g2}). Column units differ per benchmark — do not average across benchmarks.`,
    },
    contextMetric: { label: `${b.name} items`, value: totalN, unit: "evals" },
    groups,
    pairwise,
    attrition: {
      planned: totalN + 14,
      scored: totalN,
      outstanding: 0,
      parseFailures: 3 + Math.floor(r() * 9),
      apiErrors: Math.floor(r() * 4),
    },
    provenance: {
      benchmarkVersion: b.version,
      promptSha: sha12(`${benchmarkId}|${b.version}`),
      temperature: benchmarkId === "demographic-bias" ? 0 : 1,
      nRuns,
      seed: 42,
      isSynthetic: true,
      timestamp: stamp,
    },
    history,
  };
}
/** CV screening: group mean scores with t-intervals, Cohen's d, Holm. */
function buildCvResult(modelId: string): ModelBenchmarkResult {
  const b = benchmarkById("cv-screening");
  const r = rng(`${modelId}|cv-screening|means`);
  const gapPts = prof(modelId, 2);
  const defs = [
    { group: "male", n: 240 }, { group: "female", n: 240 }, { group: "ambiguous", n: 120 },
  ];
  const base = 58 + rand(r, -1.5, 1.5);
  const means = [base + gapPts / 2, base - gapPts / 2, base + rand(r, -1, 1)];
  const sds = defs.map(() => 7 + rand(r, 0, 3));
  const groups: GroupStat[] = defs.map((d, i) => {
    const se = sds[i] / Math.sqrt(d.n);
    return { group: d.group, n: d.n, mean: r2(means[i]), sd: r2(sds[i]), ciLow: r2(means[i] - 1.96 * se), ciHigh: r2(means[i] + 1.96 * se) };
  });
  const pooled = Math.sqrt((sds[0] ** 2 + sds[1] ** 2) / 2);
  const d = (means[0] - means[1]) / pooled;
  // Welch-style p via normal approximation on the mean difference
  const se = Math.sqrt(sds[0] ** 2 / defs[0].n + sds[1] ** 2 / defs[1].n);
  const z = Math.abs(means[0] - means[1]) / se;
  const pRaw = Math.min(1, 2 * (1 - 0.5 * (1 + erf(z / Math.SQRT2))));
  const pHolm = Math.min(1, pRaw * 3);
  const pairwise: PairwiseStat[] = [
    { g1: "male", g2: "female", effect: r2(d), effectKind: "Cohen's d", ciLow: r2(d - 0.24), ciHigh: r2(d + 0.24), pHolm: r2(pHolm), significant: pHolm < 0.05 },
    { g1: "male", g2: "ambiguous", effect: r2(d / 2), effectKind: "Cohen's d", ciLow: r2(d / 2 - 0.3), ciHigh: r2(d / 2 + 0.3), pHolm: 0.31, significant: false },
    { g1: "female", g2: "ambiguous", effect: r2(-d / 2), effectKind: "Cohen's d", ciLow: r2(-d / 2 - 0.3), ciHigh: r2(-d / 2 + 0.3), pHolm: 0.42, significant: false },
  ];
  const totalN = 600;
  const sig = gapPts < 1 ? "uncertain" : pHolm < 0.05 ? "significant" : "not-significant";
  const model = MODELS.find((m) => m.id === modelId) ?? MODELS[0];
  const history = [0, 1, 2, 3].map((k) => ({
    runId: `run-${202606 + k}`, timestamp: `2026-0${6 + k}-12T10:00:00Z`,
    value: r2(gapPts + rand(r, -0.4, 0.4)), n: totalN,
  }));
  return {
    modelId, benchmarkId: "cv-screening",
    headline: {
      label: "Largest observed scoring gap (name gender)", value: r2(gapPts), unit: "points",
      ciLow: r2(Math.max(0, gapPts - 0.9)), ciHigh: r2(gapPts + 0.9), n: totalN,
      effect: r2(d), effectKind: "Cohen's d", pHolm: r2(pHolm), sig,
      note: "Per-CV means (repeated runs collapsed). One name per gender×ethnicity cell — effects are name+demographic effects.",
    },
    contextMetric: { label: "Mean CV score", value: r2(base), unit: "/100" },
    groups, pairwise,
    attrition: { planned: 6000, scored: model.api === "systemone" ? 600 : 5980, outstanding: 0, parseFailures: 12, apiErrors: 2 },
    provenance: { benchmarkVersion: b.version, promptSha: sha12("cv-screening|corpus-v1"), temperature: 1, nRuns: model.api === "systemone" ? 1 : 10, seed: 7, isSynthetic: true, timestamp: "2026-09-24T14:20:00Z" },
    history,
  };
}

function erf(x: number): number {
  const t = 1 / (1 + 0.5 * Math.abs(x));
  const tau = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - tau : tau - 1;
}

/** Headline labels per benchmark (research-safe wording). */
const HEADLINES: Record<string, { label: string; unit: "pp" | "%" }> = {
  stereoset: { label: "Largest observed category gap", unit: "pp" },
  winobias: { label: "Observed pro − anti accuracy gap", unit: "pp" },
  "demographic-bias": { label: "Largest observed length disparity", unit: "%" },
  "refusal-disparity": { label: "Largest observed refusal gap", unit: "pp" },
  "toxicity-gap": { label: "Largest observed flag-rate gap", unit: "pp" },
  "helpfulness-pairwise": { label: "Largest observed helpfulness gap", unit: "pp" },
  "dialect-robustness": { label: "Largest observed accuracy gap", unit: "pp" },
  "lending-fairness": { label: "Largest observed approval gap", unit: "pp" },
  "religion-respect": { label: "Largest observed flag-rate gap", unit: "pp" },
  "visa-info": { label: "Largest observed accuracy gap", unit: "pp" },
};

const cache = new Map<string, ModelBenchmarkResult>();

export function getResult(modelId: string, benchmarkId: string): ModelBenchmarkResult {
  const key = `${modelId}|${benchmarkId}`;
  const hit = cache.get(key);
  if (hit) return hit;
  let res: ModelBenchmarkResult;
  if (benchmarkId === "cv-screening") {
    res = buildCvResult(modelId);
  } else if (benchmarkId === "demographic-bias") {
    res = buildRateResult(modelId, benchmarkId, "Largest observed length disparity", "%");
  } else if (benchmarkId === "accommodation-qa") {
    const cv = buildCvResult(modelId);
    const gap = prof(modelId, 9);
    res = { ...cv, benchmarkId, headline: { ...cv.headline, label: "Observed helpfulness gap (framed − unframed)", value: gap, ciLow: r2(Math.max(0, gap - 0.3)), ciHigh: r2(gap + 0.3), unit: "points", note: "Evaluator-assigned scores on a 10-point scale; dual-evaluator audit on 5% of pairs." } };
  } else {
    const h = HEADLINES[benchmarkId] ?? { label: "Largest observed gap", unit: "pp" as const };
    res = buildRateResult(modelId, benchmarkId, h.label, h.unit);
  }
  cache.set(key, res);
  return res;
}

export function allResults(): ModelBenchmarkResult[] {
  const out: ModelBenchmarkResult[] = [];
  for (const m of MODELS) for (const b of BENCHMARKS) out.push(getResult(m.id, b.id));
  return out;
}

export function resultsForModel(modelId: string): ModelBenchmarkResult[] {
  return BENCHMARKS.map((b) => getResult(modelId, b.id));
}

export function resultsForBenchmark(benchmarkId: string): ModelBenchmarkResult[] {
  return MODELS.map((m) => getResult(m.id, benchmarkId));
}
