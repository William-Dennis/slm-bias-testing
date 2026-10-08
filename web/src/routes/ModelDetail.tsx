/* Model detail: per-benchmark disparity panels with groups, history,
   attrition and provenance. Deliberately no aggregate fairness number. */
import { Link, useParams } from "react-router-dom";
import { BarsWithCI } from "../components/charts";
import { TrendChart } from "../components/trends";
import { Crumbs, DataTable, DemoBanner, Headline, Methodology, SigBadge, SynthBadge, Trail, fmt, type Col } from "../components/ui";
import { benchmarkById } from "../data/benchmarks";
import { formatParams, modelById } from "../data/models";
import { modelAccuracy, modelCostC, modelLatency, resultsForModel } from "../data/results";
import { experimentsForModel, modelEvalCount } from "../data/workspace";
import type { PairwiseStat } from "../data/types";

const pairCols: Col<PairwiseStat>[] = [
  { key: "pair", head: "Comparison", render: (p) => <span><strong>{p.g1}</strong> <span className="muted">vs</span> <strong>{p.g2}</strong></span> },
  { key: "eff", head: "Effect", num: true, render: (p) => <span className="num">{p.effectKind} {fmt(p.effect, 2)}</span>, sortVal: (p) => Math.abs(p.effect) },
  { key: "ci", head: "95% CI of gap", num: true, render: (p) => <span className="num muted">[{fmt(p.ciLow)}, {fmt(p.ciHigh)}]</span> },
  { key: "p", head: "p (Holm)", num: true, render: (p) => <span className="num">{p.pHolm === null ? "n/a" : p.pHolm}</span>, sortVal: (p) => p.pHolm ?? 2 },
  { key: "sig", head: "Evidence", render: (p) => (p.significant ? <SigBadge sig="significant" /> : <SigBadge sig="not-significant" />) },
];

export function ModelDetail() {
  const { id = "" } = useParams();
  const m = modelById(id);
  const results = resultsForModel(m.id);
  const exps = experimentsForModel(m.id);
  const sig = results.filter((r) => r.headline.sig === "significant").length;

  return (
    <>
      <Crumbs items={[{ label: "Models", to: "/models" }, { label: m.id }]} />
      <h1 className="page-title">{m.id} <SynthBadge /></h1>
      <p className="page-sub">
        <code>{m.ollamaTag}</code> · {formatParams(m.params)} · {m.family} · {m.architecture} · released {m.releaseDate} · <code>{m.api}</code> API.
        Disparity is reported per benchmark below — this page has no single fairness score by design.
      </p>
      <DemoBanner />

      <div className="grid c4">
        <div className="card stat-card"><div className="stat-label">Significant gaps</div><div className="stat-value">{sig}<span className="muted" style={{ fontSize: 15 }}>/12</span></div><div className="stat-meta">Holm-adjusted p &lt; 0.05</div></div>
        <div className="card stat-card"><div className="stat-label">Task accuracy</div><div className="stat-value">{fmt(modelAccuracy(m.id), 0)}%</div><div className="stat-meta">mean across task benchmarks</div></div>
        <div className="card stat-card"><div className="stat-label">Latency / cost</div><div className="stat-value small num">{modelLatency(m.id)} ms · ¢{modelCostC(m.id).toFixed(1)}/1k</div><div className="stat-meta">median over completed runs</div></div>
        <div className="card stat-card"><div className="stat-label">Evaluations</div><div className="stat-value">{modelEvalCount(m.id).toLocaleString()}</div><div className="stat-meta">{exps.length} experiments · history preserved</div></div>
      </div>

      {results.map((r, idx) => {
        const b = benchmarkById(r.benchmarkId);
        return (
          <div key={r.benchmarkId} className="card mt">
            <div className="row-flex">
              <Link to={`/benchmarks/${b.id}`} style={{ fontWeight: 650, fontSize: 15 }}>{b.name}</Link>
              {b.isSynthetic ? <SynthBadge /> : <span className="badge live">instrumented</span>}
              <span className="muted" style={{ fontSize: 12 }}>{b.category.join(" · ")}</span>
              <span className="spacer" />
              <span className="num" style={{ fontWeight: 700 }}>{fmt(r.headline.value)} {r.headline.unit}</span>
              <SigBadge sig={r.headline.sig} />
              <Link className="btn sm" to={`/tasks?benchmark=${b.id}&model=${m.id}`}>inspect pairs</Link>
            </div>
            <details open={idx < 3}>
              <summary>Analysis, history &amp; methodology</summary>
              <div className="grid c2 mt">
              <div>
                <p className="chart-q">{r.headline.label}</p>
                <Headline h={r.headline} small />
                <div className="mt"><Trail steps={[
                  { label: "Model", to: `/models/${m.id}` },
                  { label: b.name, to: `/benchmarks/${b.id}` },
                  { label: `${r.attrition.scored.toLocaleString()} scored evals` },
                ]} /></div>
              </div>
              <div>
                <p className="chart-q">Group means with 95% intervals</p>
                <p className="chart-sub">n per group shown under each bar. {r.contextMetric.label}: <span className="num">{fmt(r.contextMetric.value)} {r.contextMetric.unit}</span>.</p>
                <BarsWithCI groups={r.groups} unit={r.headline.unit} height={220} />
              </div>
            </div>
            <h2 className="section-title">Pairwise comparisons <span className="muted" style={{ fontWeight: 400 }}>(Holm-adjusted)</span></h2>
            <DataTable cols={pairCols} rows={r.pairwise} rowKey={(p) => `${p.g1}-${p.g2}`} />
            <div className="grid c2 mt">
              <div>
                <p className="chart-q">Run history</p>
                <TrendChart history={r.history} unit={r.headline.unit} height={130} />
                <p className="chart-note">Append-only: {r.history.map((h) => h.runId).join(", ")}. Superseded runs are retained, never overwritten.</p>
              </div>
              <div>
                <Methodology
                  what={`${b.name}: ${r.headline.label.toLowerCase()}.`}
                  how={b.methodology}
                  stats={`95% intervals; ${r.headline.effectKind} = ${fmt(r.headline.effect, 2)}; attrition ${r.attrition.scored.toLocaleString()}/${r.attrition.planned.toLocaleString()} scored.`}
                  limits={b.limitations}
                />
                <dl className="kv">
                  <dt>Benchmark version</dt><dd className="mono">{r.provenance.benchmarkVersion}</dd>
                  <dt>Prompt sha</dt><dd className="mono">{r.provenance.promptSha.slice(0, 12)}…</dd>
                  <dt>Runs / temperature</dt><dd className="mono">{r.provenance.nRuns} / {r.provenance.temperature}</dd>
                  <dt>Synthetic</dt><dd className="mono">is_synthetic = true</dd>
                </dl>
              </div>
            </div>
            </details>
          </div>
        );
      })}

      <h2 className="section-title">Experiments involving {m.id}</h2>
      <div className="table-wrap">
        <table className="data">
          <tbody>
            {exps.map((e) => (
              <tr key={e.id}>
                <td><Link to={`/experiments/${e.id}`} className="cell-main">{e.name}</Link><div className="cell-sub">v{e.runVersion}{e.supersedes ? ` · supersedes ${e.supersedes}` : ""}</div></td>
                <td><span className="badge">{e.status}</span></td>
                <td className="n num muted">{e.estimates.evals.toLocaleString()} evals</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
