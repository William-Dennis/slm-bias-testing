/* Pair detail: condition A vs B with full evaluator provenance and an
   evidence trail back to experiment → benchmark → model. */
import { Link, useParams } from "react-router-dom";
import { Crumbs, DemoBanner, SynthBadge, Trail, fmt } from "../components/ui";
import { benchmarkById } from "../data/benchmarks";
import { modelById } from "../data/models";
import { pairById } from "../data/workspace";
import type { JudgeEval } from "../data/types";

function EvalCard({ e, title }: { e: JudgeEval; title: string }) {
  return (
    <div className="card mt">
      <p className="chart-q">{title} — evaluator-assigned scores</p>
      <p className="chart-sub">{e.judgeModel} {e.judgeVersion} · shown {e.orderingShown} · seed {e.seed} · confidence {e.confidence}</p>
      <dl className="kv">
        <dt>Rubric</dt><dd>{e.rubric}</dd>
        <dt>Score A / B</dt><dd className="num">{e.scoreA} / {e.scoreB}</dd>
        <dt>Temperature</dt><dd className="num">{e.temperature}</dd>
        <dt>Rationale</dt><dd>{e.rationale}</dd>
      </dl>
      <p className="chart-note">Evaluator output is an assigned score, not ground truth. Ordering shown to the judge is preserved above.</p>
    </div>
  );
}

export function TaskDetail() {
  const { id = "" } = useParams();
  const p = pairById(id);
  if (!p) return (<><h1 className="page-title">Pair not found</h1><p className="page-sub"><code>{id}</code> is unknown. <Link to="/tasks">Back to Task Explorer</Link>.</p></>);
  const b = benchmarkById(p.benchmarkId);
  const m = modelById(p.modelId);

  return (
    <>
      <Crumbs items={[{ label: "Tasks", to: "/tasks" }, { label: p.id }]} />
      <h1 className="page-title mono" style={{ fontSize: 17 }}>{p.id} <SynthBadge /></h1>
      <p className="page-sub">{p.task} · changed variable: <strong>{p.changedVariable}</strong> ({p.baselineValue} → {p.counterfactualValue}) · shown {p.ordering}{p.randomized ? ` · randomized (seed ${p.seed})` : ""}</p>
      <DemoBanner />

      {p.judgeDisagreement ? <div className="demo-banner" style={{ borderColor: "rgba(229,181,103,.3)" }}><strong>Evaluators disagree</strong><span>Two evaluator records below assign different directions. Both are retained — do not average them.</span></div> : null}
      {p.failed ? <div className="demo-banner"><strong>Evaluation failed</strong><span>No score was produced for this pair; it counts in attrition, not in metrics.</span></div> : null}
      {p.refusal ? <div className="demo-banner"><strong>Refusal present</strong><span>One condition refused; scored as 0 for gap purposes and flagged for review.</span></div> : null}

      <div className="pair-grid">
        <div className="cond">
          <div className="cond-head"><span className="cond-tag">CONDITION A</span><span className="muted" style={{ fontSize: 12 }}>{p.baselineValue}</span><span className="spacer" /><span className="num" style={{ fontWeight: 700 }}>{p.scoreA === null ? "n/a" : fmt(p.scoreA)}</span></div>
          <div className="muted" style={{ fontSize: 11 }}>PROMPT</div>
          <div className="prompt-box">{p.promptA}</div>
          <div className="muted" style={{ fontSize: 11 }}>RESPONSE</div>
          <div className="response-box">{p.responseA || "(empty)"}</div>
        </div>
        <div className="cond">
          <div className="cond-head"><span className="cond-tag b">CONDITION B</span><span className="muted" style={{ fontSize: 12 }}>{p.counterfactualValue}</span><span className="spacer" /><span className="num" style={{ fontWeight: 700 }}>{p.scoreB === null ? "n/a" : fmt(p.scoreB)}</span></div>
          <div className="muted" style={{ fontSize: 11 }}>PROMPT</div>
          <div className="prompt-box">{p.promptB}</div>
          <div className="muted" style={{ fontSize: 11 }}>RESPONSE</div>
          <div className="response-box">{p.responseB || "(empty)"}</div>
        </div>
      </div>

      <div className="card mt">
        <div className="row-flex">
          <span className="stat-label">Paired difference (A − B)</span>
          <span className="stat-value small num">{p.scoreDiff === null ? "n/a" : `${p.scoreDiff > 0 ? "+" : ""}${fmt(p.scoreDiff)}`}</span>
          {p.outlier ? <span className="badge">outlier</span> : null}
          {p.stereotypeExample ? <span className="badge">stereotype example</span> : null}
        </div>
        <div className="mt"><Trail steps={[
          { label: "Experiment", to: `/experiments/${p.experimentId}` },
          { label: b.name, to: `/benchmarks/${b.id}` },
          { label: m.id, to: `/models/${m.id}` },
          { label: p.id },
        ]} /></div>
      </div>

      {p.evaluator ? <EvalCard e={p.evaluator} title="Primary evaluator" /> : null}
      {p.secondEvaluator ? <EvalCard e={p.secondEvaluator} title="Second evaluator" /> : null}
    </>
  );
}
