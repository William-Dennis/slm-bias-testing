/* Benchmark detail: research question, methodology, cross-model
   results with intervals, example pairs, limitations, evidence links. */
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { BarsWithCI } from "../components/charts";
import { Crumbs, DataTable, DemoBanner, Headline, Methodology, SigBadge, SynthBadge, Trail, fmt, type Col } from "../components/ui";
import { benchmarkById } from "../data/benchmarks";
import { MODELS, modelById } from "../data/models";
import { getResult } from "../data/results";
import { EXPERIMENTS, experimentsForBenchmark, pairById } from "../data/workspace";
import { pairsForExperiment } from "../data/pairs";

interface Row { id: string; value: number; unit: string; ci: string; n: number; effect: string; sig: "significant" | "not-significant" | "uncertain" }

export function BenchmarkDetail() {
  const { id = "" } = useParams();
  const b = benchmarkById(id);
  const [focus, setFocus] = useState("smollm2-135m");
  const focusRes = getResult(MODELS.some((m) => m.id === focus) ? focus : MODELS[0].id, b.id);
  const exps = experimentsForBenchmark(b.id);

  const rows: Row[] = useMemo(() => MODELS.map((m) => {
    const r = getResult(m.id, b.id);
    return {
      id: m.id, value: r.headline.value, unit: r.headline.unit,
      ci: `[${fmt(r.headline.ciLow)}, ${fmt(r.headline.ciHigh)}]`, n: r.headline.n,
      effect: `${r.headline.effectKind} ${fmt(r.headline.effect, 2)}`, sig: r.headline.sig,
    };
  }).sort((a, z) => z.value - a.value), [b.id]);

  const cols: Col<Row>[] = [
    { key: "id", head: "Model", render: (r) => (<><span className="cell-main"><Link to={`/models/${r.id}`}>{r.id}</Link></span><div className="cell-sub">{modelById(r.id).family}</div></>), sortVal: (r) => r.id },
    { key: "v", head: `Observed gap (${b.metrics[0].unit})`, num: true, render: (r) => <span className="num">{fmt(r.value)} {r.unit === "%" ? "%" : r.unit}</span>, sortVal: (r) => r.value },
    { key: "ci", head: "95% CI", num: true, render: (r) => <span className="num muted">{r.ci}</span> },
    { key: "n", head: "n", num: true, render: (r) => <span className="num">{r.n.toLocaleString()}</span>, sortVal: (r) => r.n },
    { key: "e", head: "Effect", num: true, render: (r) => <span className="num muted">{r.effect}</span> },
    { key: "s", head: "Evidence", render: (r) => <SigBadge sig={r.sig} /> },
  ];

  const exampleExp = EXPERIMENTS.find((e) => e.benchmarkId === b.id);
  const examples = exampleExp ? pairsForExperiment(exampleExp.id, exampleExp.modelIds[0], b.id, 24).filter((p) => !p.failed).slice(0, 2) : [];

  return (
    <>
      <Crumbs items={[{ label: "Benchmarks", to: "/benchmarks" }, { label: b.name }]} />
      <h1 className="page-title">{b.name} {b.isSynthetic ? <SynthBadge /> : <span className="badge live">instrumented</span>}</h1>
      <p className="page-sub">{b.description}</p>
      <DemoBanner />

      <div className="card">
        <p className="chart-q">Research question</p>
        <p style={{ fontSize: 14 }}>{b.researchQuestion}</p>
        <Methodology
          what={`${b.metrics[0].label} (${b.metrics[0].unit}). ${b.metrics[0].directionNote}`}
          how={b.methodology}
          stats="Group means with 95% intervals; pairwise effects with Holm–Bonferroni correction; attrition tracked per run."
          limits={b.limitations}
        />
        <dl className="kv">
          <dt>Controlled variables</dt><dd>{b.controlledVars.join("; ")}</dd>
          <dt>Tasks</dt><dd className="num">{b.tasks.toLocaleString()} · variables: {b.variablesTested.join("; ")}</dd>
          <dt>Version</dt><dd className="mono">{b.version} · is_synthetic = {String(true)}</dd>
        </dl>
      </div>

      <h2 className="section-title">Model comparison <span className="muted" style={{ fontWeight: 400 }}>— which models show the largest observed gap?</span></h2>
      <DataTable cols={cols} rows={rows} rowKey={(r) => r.id} linkTo={(r) => `/models/${r.id}`} />

      <div className="grid c2 mt">
        <div className="card">
          <div className="toolbar">
            <label className="f-label" htmlFor="focus">Focus model</label>
            <select id="focus" className="input" value={focus} onChange={(e) => setFocus(e.target.value)}>
              {MODELS.map((m) => <option key={m.id} value={m.id}>{m.id}</option>)}
            </select>
          </div>
          <Headline h={focusRes.headline} small />
          <div className="mt"><Trail steps={[{ label: b.name }, { label: focusRes.modelId, to: `/models/${focusRes.modelId}` }, { label: "pairs", to: `/tasks?benchmark=${b.id}&model=${focusRes.modelId}` }]} /></div>
        </div>
        <div className="card">
          <p className="chart-q">Group means with 95% intervals — {focusRes.modelId}</p>
          <BarsWithCI groups={focusRes.groups} unit={focusRes.headline.unit} />
        </div>
      </div>

      <h2 className="section-title">Example cases</h2>
      {examples.map((p) => {
        const full = pairById(p.id);
        return (
          <div key={p.id} className="card mb">
            <div className="row-flex">
              <Link to={`/tasks/${p.id}`} className="cell-main mono">{p.id}</Link>
              <span className="badge">Δ {p.scoreDiff === null ? "n/a" : fmt(p.scoreDiff)} </span>
              <span className="muted" style={{ fontSize: 12 }}>changed: {p.changedVariable}</span>
              <span className="spacer" />
              <Link className="btn sm" to={`/tasks/${p.id}`}>open pair</Link>
            </div>
            <p className="chart-note">{p.task} · {full ? `evaluator: ${full.evaluator?.judgeModel ?? "regex parse"}${full.judgeDisagreement ? " · evaluators disagree" : ""}` : ""}</p>
          </div>
        );
      })}

      <h2 className="section-title">Experiments on this benchmark</h2>
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
