/* Deterministic PRNG + real interval/effect-size calculations.
   Statistics shown in the UI are computed here from generated counts —
   the same estimators the Python package uses (Wilson, Newcombe,
   Cohen's h/d, Holm-Bonferroni). */

// mulberry32 — deterministic per seed string
export function rng(seed: string): () => number {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  let a = h >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const rand = (r: () => number, lo: number, hi: number): number =>
  lo + r() * (hi - lo);

export function pick<T>(r: () => number, arr: T[]): T {
  return arr[Math.floor(r() * arr.length)];
}

/** Wilson 95% interval for a proportion k/n (as percentages). */
export function wilson(k: number, n: number, z = 1.96): [number, number] {
  if (n <= 0) return [0, 0];
  const p = k / n;
  const denom = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denom;
  const half =
    (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, (centre - half) * 100), Math.min(100, (centre + half) * 100)];
}

/** Newcombe 95% interval for a difference of two proportions (pp). */
export function newcombe(
  k1: number, n1: number, k2: number, n2: number, z = 1.96
): [number, number] {
  const p1 = n1 > 0 ? k1 / n1 : 0;
  const p2 = n2 > 0 ? k2 / n2 : 0;
  const diff = (p1 - p2) * 100;
  const [, u1] = wilson(k1, n1, z);
  const [l1] = wilson(k1, n1, z);
  const [, u2] = wilson(k2, n2, z);
  const [l2] = wilson(k2, n2, z);
  const lo = diff - Math.sqrt(Math.pow(p1 * 100 - l1, 2) + Math.pow(u2 - p2 * 100, 2));
  const hi = diff + Math.sqrt(Math.pow(u1 - p1 * 100, 2) + Math.pow(p2 * 100 - l2, 2));
  return [lo, hi];
}

/** Cohen's h for two proportions. */
export function cohensH(p1: number, p2: number): number {
  const c = (p: number) => 2 * Math.asin(Math.sqrt(Math.min(0.9999, Math.max(0.0001, p))));
  return c(p1) - c(p2);
}

/** Two-proportion z-test p-value (two-sided). */
export function twoPropP(k1: number, n1: number, k2: number, n2: number): number {
  if (n1 < 2 || n2 < 2) return 1;
  const p = (k1 + k2) / (n1 + n2);
  const se = Math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2));
  if (se === 0) return 1;
  const z = Math.abs(k1 / n1 - k2 / n2) / se;
  // Normal survival via erf approximation
  const t = 1 / (1 + 0.2316419 * z);
  const d = 0.3989423 * Math.exp((-z * z) / 2);
  const tail =
    d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return Math.min(1, 2 * tail);
}

/** Holm-Bonferroni adjustment over a list of raw p-values. */
export function holm(pvals: number[]): (number | null)[] {
  const order = pvals
    .map((p, i) => ({ p, i }))
    .filter((x) => Number.isFinite(x.p))
    .sort((a, b) => a.p - b.p);
  const m = order.length;
  const adj = new Array<number | null>(pvals.length).fill(null);
  let running = 0;
  order.forEach((x, rank) => {
    running = Math.max(running, Math.min(1, (m - rank) * x.p));
    adj[x.i] = running;
  });
  return adj;
}

export const r2 = (x: number): number => Math.round(x * 100) / 100;
