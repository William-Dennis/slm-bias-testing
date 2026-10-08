/* Task Explorer: every counterfactual pair, filterable to the cases
   that need a researcher — disagreements, refusals, failures, outliers. */
import { useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { BENCHMARKS } from "../data/benchmarks";
import { MODELS } from "../data/models";
import { pairsForExperiment } from "../data/pairs";
import type { CounterfactualPair } from "../data/types";
import { EXPERIMENTS, allPairs } from "../data/workspace";
import { DataTable, DemoBanner, SynthBadge, fmt, type Col } from "../components/ui";

type Flag = "all" | "largest" | "disagreement" | "refusal" | "failed" | "stereotype" | "outlier";

function param(p: URLSearchParams, k: string, fb: string): string {
  return p.get(k) ?? fb;
}

export function Tasks() {
  const [params, setParams] = useSearchParams();
  const bench = param(params, "benchmark", "all");
  const model = param(params, "model", "all");
  const exp = param(params, "experiment", "all");
  const flag = param(params, "flag", "all") as Flag;
  const set = (k: string, v: string) => {
    const next = new URLSearchParams(params);
    if (v === "all") next.delete(k);
    else next.set(k, v);
    setParams(next, { replace: true });
  };

  const pairs = useMemo(() => {
    let list: CounterfactualPair[];
    if (exp !== "all") {
      const e = EXPERIMENTS.find((x) => x.id === exp);
      if (!e) return [];
      const ms = model === "all" ? e.modelIds : e.modelIds.filter((m) => m === model);
      list = ms.flatMap((m) => pairsForExperiment(e.id, m, e.benchmarkId, 24));
    } else {
      list = allPairs();
    }
    if (bench !== "all") list = list.filter((p) => p.benchmarkId === bench);
    if (model !== "all" && exp === "all") list = list.filter((p) => p.modelId === model);
    switch (flag) {
      case "disagreement": return list.filter((p) => p.judgeDisagreement);
      case "refusal": return list.filter((p) => p.refusal);
      case "failed": return list.filter((p) => p.failed);
      case "stereotype": return list.filter((p) => p.stereotypeExample);
      case "outlier": return list.filter((p) => p.outlier);
      case "largest": return [...list].filter((p) => p.scoreDiff !== null).sort((a, b) => Math.abs(b.scoreDiff ?? 0) - Math.abs(a.scoreDiff ?? 0)).slice(0, 60);
      default: return list.slice(0, 120);
    }
  }, [bench, model, exp, flag]);

  const cols: Col<CounterfactualPair>[] = [
    { key: "id", head: "Pair", render: (p) => (<><span className="cell-main mono" style={{ fontSize: 12 }}><Link to={`/tasks/${p.id}`}>{p.id}</Link></span><div className="cell-sub">{p.task} · {p.modelId}</div></>), sortVal: (p) => p.id },
    { key: "chg", head: "Changed variable", render: (p) => (<span style={{ fontSize: 12 }}>{p.changedVariable}<div className="cell-sub">{p.baselineValue} → {p.counterfactualValue}</div></span>) },
    { key: "diff", head: "Paired Δ", num: true, render: (p) => <span className="num">{p.scoreDiff === null ? "n/a" : `${p.scoreDiff > 0 ? "+" : ""}${fmt(p.scoreDiff)}`}</span>, sortVal: (p) => Math.abs(p.scoreDiff ?? -1) },
    { key: "eval", head: "Evaluator", render: (p) => (<span style={{ fontSize: 12 }}>{p.evaluator ? <>{p.evaluator.judgeModel}<div className="cell-sub">conf {p.evaluator.confidence} · {p.evaluator.orderingShown}</div></> : <span className="muted">regex parse</span>}</span>) },
    { key: "flags", head: "Flags", render: (p) => (<span className="row-flex">{p.judgeDisagreement ? <span className="badge uncertain">disagreement</span> : null}{p.refusal ? <span className="badge">refusal</span> : null}{p.failed ? <span className="badge sig">failed</span> : null}{p.outlier ? <span className="badge">outlier</span> : null}{p.stereotypeExample ? <span className="badge">stereotype</span> : null}</span>) },
  ];

  return (
    <>
      <h1 className="page-title">Task Explorer</h1>
      <p className="page-sub">Counterfactual pairs side by side — both prompts, both responses, the changed variable, and every evaluator record. Start from flags, end at evidence. <SynthBadge /></p>
      <DemoBanner />
      <div className="toolbar">
        <select className="input" value={bench} onChange={(e) => set("benchmark", e.target.value)}>
          <option value="all">All benchmarks</option>
          {BENCHMARKS.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}
        </select>
        <select className="input" value={model} onChange={(e) => set("model", e.target.value)}>
          <option value="all">All models</option>
          {MODELS.map((m) => <option key={m.id} value={m.id}>{m.id}</option>)}
        </select>
        <select className="input" value={exp} onChange={(e) => set("experiment", e.target.value)}>
          <option value="all">All experiments</option>
          {EXPERIMENTS.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
        </select>
        <select className="input" value={flag} onChange={(e) => set("flag", e.target.value)}>
          <option value="all">All pairs</option>
          <option value="largest">Largest disparities</option>
          <option value="disagreement">Judge disagreement</option>
          <option value="refusal">Refusals</option>
          <option value="failed">Failed evaluations</option>
          <option value="stereotype">Stereotype examples</option>
          <option value="outlier">Outliers</option>
        </select>
        <span className="muted">{pairs.length} pairs</span>
      </div>
      {pairs.length === 0
        ? <div className="empty">No pairs match these filters. Try widening the benchmark or clearing the flag.</div>
        : <DataTable cols={cols} rows={pairs} rowKey={(p) => p.id} linkTo={(p) => `/tasks/${p.id}`} />}
    </>
  );
}
