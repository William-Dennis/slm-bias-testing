/* Experiment detail: status, configuration, per-model results with
   intervals, run log, version history, and the path to raw pairs. */
import { Link, useParams } from "react-router-dom";
import { BarsWithCI } from "../components/charts";
import { Crumbs, DataTable, DemoBanner, Headline, Methodology, SigBadge, SynthBadge, Trail, fmt, type Col } from "../components/ui";
import { benchmarkById } from "../data/benchmarks";
import { modelById } from "../data/models";
import { getResult } from "../data/results";
import { JUDGES, allExperiments } from "../data/workspace";

interface Row { id: string; value: number; unit: string; ci: string; n: number; sig: "significant" | "not-significant" | "uncertain" }

export function ExperimentDetail() {
  const { id = "" } = useParams();
  const e = allExperiments().find((x) => x.id === id);
  if (!e) return (<><h1 className="page-title">Experiment not found</h1><p className="page-sub">Unknown id <code>{id}</code>. <Link to="/experiments">Back to experiments</Link>.</p></>);
  const b = benchmarkById(e.benchmarkId);
  const judge = JUDGES.find((j) => j.id === e.config.judgeId);

  const rows: Row[] = e.modelIds.map((mid) => {
    const r = getResult(mid, e.benchmarkId);
    return { id: mid, value: r.headline.value, unit: r.headline.unit, ci: `[${fmt(r.headline.ciLow)}, ${fmt(r.headline.ciHigh)}]`, n: r.headline.n, sig: r.headline.sig };
  });

  const cols: Col<Row>[] = [
    { key: "id", head: "Model", render: (r) => (<><span className="cell-main"><Link to={`/models/${r.id}`}>{r.id}</Link></span><div className="cell-sub">{modelById(r.id).family}</div></>), sortVal: (r) => r.id },
    { key: "v", head: `Observed gap (${b.metrics[0].unit})`, num: true, render: (r) => <span className="num">{fmt(r.value)} {r.unit === "%" ? "%" : r.unit}</span>, sortVal: (r) => r.value },
    { key: "ci", head: "95% CI", num: true, render: (r) => <span className="num muted">{r.ci}</span> },
    { key: "n", head: "n", num: true, render: (r) => <span className="num">{r.n.toLocaleString()}</span>, sortVal: (r) => r.n },
    { key: "s", head: "Evidence", render: (r) => <SigBadge sig={r.sig} /> },
    { key: "pairs", head: "Pairs", render: (r) => <Link className="btn sm" to={`/tasks?experiment=${e.id}&model=${r.id}`}>pairs</Link> },
  ];

  const first = getResult(e.modelIds[0], e.benchmarkId);

  return (
    <>
      <Crumbs items={[{ label: "Experiments", to: "/experiments" }, { label: e.name }]} />
      <h1 className="page-title">{e.name} <SynthBadge /></h1>
      <p className="page-sub">
        <Link to={`/benchmarks/${b.id}`}>{b.name}</Link> · v{e.runVersion}
        {e.supersedes ? <> · supersedes <code>{e.supersedes}</code> (retained)</> : null} ·{" "}
        <span className="badge"><span className={`dot ${e.status === "complete" ? "ok" : "warn pulse"}`} />{e.status}{e.status !== "complete" ? ` ${e.progress}%` : ""}</span>
      </p>
      <DemoBanner />

      <div className="grid c4">
        <div className="card stat-card"><div className="stat-label">Models</div><div className="stat-value">{e.modelIds.length}</div><div className="stat-meta">{e.modelIds.join(", ")}</div></div>
        <div className="card stat-card"><div className="stat-label">Evaluations</div><div className="stat-value">{e.estimates.evals.toLocaleString()}</div><div className="stat-meta">planned; attrition tracked per model</div></div>
        <div className="card stat-card"><div className="stat-label">Runtime / cost</div><div className="stat-value small num">≈{e.estimates.runtimeMin} min · ${e.estimates.costUsd.toFixed(2)}</div><div className="stat-meta">estimates {e.completedAt ? "· completed " + e.completedAt.slice(0, 10) : "· in progress"}</div></div>
        <div className="card stat-card"><div className="stat-label">Judge</div><div className="stat-value small">{judge?.model ?? e.config.judgeId}</div><div className="stat-meta">{judge?.version} · evaluator, not ground truth</div></div>
      </div>

      <h2 className="section-title">Results by model</h2>
      <DataTable cols={cols} rows={rows} rowKey={(r) => r.id} />

      <div className="grid c2 mt">
        <div className="card">
          <p className="chart-q">{first.headline.label} — {first.modelId}</p>
          <Headline h={first.headline} small />
          <div className="mt"><Trail steps={[{ label: "Experiment" }, { label: b.name, to: `/benchmarks/${b.id}` }, { label: first.modelId, to: `/models/${first.modelId}` }, { label: "pairs", to: `/tasks?experiment=${e.id}&model=${first.modelId}` }]} /></div>
        </div>
        <div className="card">
          <p className="chart-q">Group means — {first.modelId}</p>
          <BarsWithCI groups={first.groups} unit={first.headline.unit} height={220} />
        </div>
      </div>

      <div className="grid c2 mt">
        <div className="card">
          <p className="chart-q">Configuration</p>
          <dl className="kv">
            <dt>Trials</dt><dd className="num">{e.config.trials}</dd>
            <dt>Temperature</dt><dd className="num">{e.config.temperature}</dd>
            <dt>System prompt</dt><dd className="mono">{e.config.systemPrompt}</dd>
            <dt>Eval method</dt><dd className="mono">{e.config.evalMethod}</dd>
            <dt>Randomization</dt><dd className="mono">{e.config.randomization ? `on (seed ${e.config.seed})` : "off"}</dd>
            <dt>Judge prompt</dt><dd className="mono" style={{ fontSize: 11 }}>{judge?.promptTemplate.slice(0, 120)}…</dd>
          </dl>
          <Methodology what={`${b.name}: ${b.researchQuestion}`} how={b.methodology} stats="Intervals and Holm-adjusted comparisons per model panel; see model pages for full tables." limits={b.limitations} />
        </div>
        <div className="card">
          <p className="chart-q">Run log</p>
          <div className="log-box">{e.log.map((l, i) => <div key={i}>{l}</div>)}</div>
          <p className="chart-note">Provenance: <span className="mono">{b.version}</span> · seed <span className="mono">{e.config.seed}</span> · is_synthetic = true · created {e.createdAt.slice(0, 10)}.</p>
        </div>
      </div>
    </>
  );
}
