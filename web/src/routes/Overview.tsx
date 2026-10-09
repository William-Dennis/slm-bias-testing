/* Overview dashboard: fleet status, per-benchmark leaderboard, full
   heatmap, capability/disparity trade-off, history, recent work. */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { HeatTable, type HeatCell } from "../components/charts";
import { ScatterPlot } from "../components/trends";
import { TrendChart } from "../components/trends";
import { DataTable, DemoBanner, Headline, SigBadge, SynthBadge, Trail, fmt, type Col } from "../components/ui";
import { BENCHMARKS } from "../data/benchmarks";
import { MODELS, formatParams, modelById } from "../data/models";
import { getResult } from "../data/results";
import { EXPERIMENTS, INSIGHTS, totalEvals } from "../data/workspace";

const SHORT: Record<string, string> = {
  "cv-screening": "CV", stereoset: "STEREO", winobias: "WINO",
  "demographic-bias": "DEMO", "refusal-disparity": "REFUSAL", "toxicity-gap": "TOXIC",
  "helpfulness-pairwise": "HELP", "dialect-robustness": "DIALECT", "lending-fairness": "LEND",
  "accommodation-qa": "ACCOM", "religion-respect": "RELIG", "visa-info": "VISA",
};

const FAMILY_COLORS: Record<string, string> = {
  huggingface: "#6ea8fe", alibaba: "#e5b567", google: "#4fd1a5",
  ibm: "#a78bfa", liquid: "#7ad0e0", meta: "#e57a8b",
  stability: "#9aa7b8", convai: "#7ae0c3", bespoke: "#f0a35e",
};

interface Row {
  id: string;
  value: number;
  unit: string;
  ci: string;
  n: number;
  effect: string;
  sig: "significant" | "not-significant" | "uncertain";
  history: number[];
}

export function Overview() {
  const [bench, setBench] = useState("cv-screening");
  const b = BENCHMARKS.find((x) => x.id === bench) ?? BENCHMARKS[0];

  const rows: Row[] = useMemo(
    () => MODELS.map((m) => {
      const r = getResult(m.id, bench);
      return {
        id: m.id, value: r.headline.value, unit: r.headline.unit,
        ci: `[${fmt(r.headline.ciLow)}, ${r.headline.ciHigh}]`,
        n: r.headline.n, effect: `${r.headline.effectKind} ${fmt(r.headline.effect, 2)}`,
        sig: r.headline.sig, history: r.history.map((h) => h.value),
      };
    }).sort((a, z) => z.value - a.value),
    [bench]
  );

  const cols: Col<Row>[] = [
    { key: "model", head: "Model", render: (r) => (<><span className="cell-main"><Link to={`/models/${r.id}`}>{r.id}</Link></span><div className="cell-sub">{modelById(r.id).family} · {formatParams(modelById(r.id).params)}</div></>), sortVal: (r) => r.id },
    { key: "value", head: `Observed disparity (${b.metrics[0].unit})`, num: true, render: (r) => <span className="num">{fmt(r.value)} {r.unit === "%" ? "%" : r.unit}</span>, sortVal: (r) => r.value },
    { key: "ci", head: "95% CI", num: true, render: (r) => <span className="num muted">{r.ci}</span> },
    { key: "n", head: "n", num: true, render: (r) => <span className="num">{r.n.toLocaleString()}</span>, sortVal: (r) => r.n },
    { key: "effect", head: "Effect", num: true, render: (r) => <span className="num muted">{r.effect}</span> },
    { key: "sig", head: "Evidence", render: (r) => <SigBadge sig={r.sig} /> },
  ];

  const sigCount = useMemo(
    () => MODELS.filter((m) => getResult(m.id, bench).headline.sig === "significant").length,
    [bench]
  );

  const cells: HeatCell[][] = useMemo(
    () => [...MODELS].sort((a, c) => a.id.localeCompare(c.id)).map((m) =>
      BENCHMARKS.map((bb) => {
        const r = getResult(m.id, bb.id);
        return { value: r.headline.value, n: r.headline.n, sig: r.headline.sig, note: `${r.headline.label}: ${fmt(r.headline.value)} ${r.headline.unit} ${fmt(r.headline.ciLow, 1)}–${fmt(r.headline.ciHigh, 1)}` };
      })
    ),
    []
  );

  const scatter = useMemo(() => {
    const pts = MODELS.map((m) => {
      const r = getResult(m.id, bench);
      return {
        x: m.params / 1e6, y: r.headline.value, label: m.id,
        color: FAMILY_COLORS[m.family] ?? "#9aa7b8",
        title: `${m.id}: ${m.params / 1e6}M params, disparity ${fmt(r.headline.value)} ${r.headline.unit} (n=${r.headline.n})`,
      };
    });
    // Label only the notable points; the rest stay hoverable dots.
    const top = [...pts].sort((a, b) => b.y - a.y).slice(0, 3).map((p) => p.label);
    const biggest = [...pts].sort((a, b) => b.x - a.x)[0].label;
    return pts.map((p) => ({ ...p, showLabel: top.includes(p.label) || p.label === biggest }));
  }, [bench]);

  const featured = ["smollm-135m", "qwen25-05b", "gemma3-270m", "nimble"];

  return (
    <>
      <h1 className="page-title">Overview</h1>
      <p className="page-sub">
        Fleet-wide observed disparities across {BENCHMARKS.length} benchmarks and {MODELS.length} models.
        Every figure shows its sample size and uncertainty; <strong>no cross-benchmark averages are computed</strong>.
      </p>
      <DemoBanner />

      <div className="grid c4">
        <div className="card stat-card"><div className="stat-label">Models evaluated</div><div className="stat-value">20</div><div className="stat-meta">7 families · chat + systemone APIs</div></div>
        <div className="card stat-card"><div className="stat-label">Benchmarks</div><div className="stat-value">12</div><div className="stat-meta">4 instrumented · 8 synthetic pilots <SynthBadge /></div></div>
        <div className="card stat-card"><div className="stat-label">Evaluations</div><div className="stat-value">{totalEvals().toLocaleString()}</div><div className="stat-meta">{EXPERIMENTS.filter((e) => e.status === "complete").length} complete experiments · history preserved</div></div>
        <div className="card stat-card"><div className="stat-label">Significant gaps · {b.name}</div><div className="stat-value">{sigCount}<span className="muted" style={{ fontSize: 15 }}>/20</span></div><div className="stat-meta">Holm-adjusted p &lt; 0.05</div></div>
      </div>

      <h2 className="section-title">Model leaderboard</h2>
      <p className="section-sub">Which models show the largest observed disparity on one benchmark? Sorted by headline gap — the interval matters more than the rank.</p>
      <div className="toolbar">
        <label className="f-label" htmlFor="lb-bench">Benchmark</label>
        <select id="lb-bench" className="input" value={bench} onChange={(e) => setBench(e.target.value)}>
          {BENCHMARKS.map((x) => <option key={x.id} value={x.id}>{x.name}{x.isSynthetic ? " (synthetic)" : ""}</option>)}
        </select>
        <span className="hint">{b.researchQuestion}</span>
      </div>
      <DataTable cols={cols} rows={rows} rowKey={(r) => r.id} linkTo={(r) => `/models/${r.id}`} />
      <h2 className="section-title">Bias category heatmap</h2>
      <p className="section-sub">Every model × benchmark headline gap. Rows alphabetical — not ranked, because columns have incompatible units.</p>
      <div className="card">
        <HeatTable
          rows={[...MODELS].sort((a, c) => a.id.localeCompare(c.id)).map((m) => ({ id: m.id, label: m.id }))}
          cols={BENCHMARKS.map((x) => ({ id: x.id, label: SHORT[x.id] ?? x.id, full: x.name }))}
          cells={cells}
          unitNote="Each cell is that benchmark's own headline gap (pp, %, points or /100 — see column)."
        />
      </div>

      <div className="grid c2 mt">
        <div className="card">
          <p className="chart-q">Does scale trade off against disparity?</p>
          <p className="chart-sub">Parameters vs observed disparity on {b.name}. No trend is fitted — points only, because n = 20 with family confounding.</p>
          <ScatterPlot pts={scatter} xLabel="Parameters (M)" yLabel={`Observed disparity (${b.metrics[0].unit})`} />
          <p className="chart-note">Colour = model family. Hover a point for n and interval.</p>
        </div>
        <div className="card">
          <p className="chart-q">How have headline gaps moved across runs?</p>
          <p className="chart-sub">Run history is append-only; superseded runs stay visible. No trend line is drawn — four points cannot support one.</p>
          {featured.map((id) => {
            const r = getResult(id, bench);
            return (
              <div key={id} className="mb">
                <div className="row-flex"><Link to={`/models/${id}`}><strong>{id}</strong></Link><SynthBadge /><span className="muted num">n={r.headline.n.toLocaleString()}/run</span></div>
                <TrendChart history={r.history} unit={r.headline.unit} height={120} />
              </div>
            );
          })}
        </div>
      </div>

      <div className="grid c2 mt">
        <div>
          <h2 className="section-title" style={{ marginTop: 0 }}>Recent experiments</h2>
          <div className="table-wrap">
            <table className="data">
              <tbody>
                {EXPERIMENTS.slice(0, 5).map((e) => (
                  <tr key={e.id}>
                    <td><Link to={`/experiments/${e.id}`} className="cell-main">{e.name}</Link><div className="cell-sub">{e.modelIds.length} models · v{e.runVersion}{e.supersedes ? ` · supersedes ${e.supersedes.slice(0, 13)}…` : ""}</div></td>
                    <td className="n"><span className="badge">{e.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h2 className="section-title" style={{ marginTop: 0 }}>Recent findings</h2>
          {INSIGHTS.filter((i) => i.kind === "measured").slice(0, 3).map((i) => (
            <div key={i.id} className="card mb">
              <div className="insight-kind measured">Measured result</div>
              <div className="mt" style={{ fontWeight: 600 }}>{i.title}</div>
              <p className="chart-note">{i.body.split("Interpretation:")[0]}</p>
              <div className="row-flex">{i.links.slice(0, 2).map((l) => <Link key={l.to} className="btn sm" to={l.to}>{l.label}</Link>)}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card mt">
        <p className="chart-q">Largest observed gap fleet-wide — with context</p>
        <Headline h={getResult(rows[0].id, bench).headline} small />
        <div className="mt"><Trail steps={[{ label: "Overview", to: "/" }, { label: b.name, to: `/benchmarks/${bench}` }, { label: rows[0].id, to: `/models/${rows[0].id}` }, { label: "pairs" }]} /></div>
      </div>
    </>
  );
}
