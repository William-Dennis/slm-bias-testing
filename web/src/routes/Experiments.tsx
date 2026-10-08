/* Experiments: append-only run history. Superseded runs stay listed —
   nothing is silently replaced. */
import { Link } from "react-router-dom";
import { useState } from "react";
import { benchmarkById } from "../data/benchmarks";
import { allExperiments } from "../data/workspace";
import { DataTable, DemoBanner, SynthBadge, type Col } from "../components/ui";
import type { Experiment } from "../data/types";

export function Experiments() {
  const [only, setOnly] = useState("all");
  // Re-read on each render so newly completed runs appear.
  const exps = allExperiments().filter((e) => only === "all" || e.status === only);

  const cols: Col<Experiment>[] = [
    { key: "name", head: "Experiment", render: (e) => (<><span className="cell-main"><Link to={`/experiments/${e.id}`}>{e.name}</Link></span><div className="cell-sub">v{e.runVersion}{e.supersedes ? ` · supersedes ${e.supersedes.slice(0, 18)}` : ""} · {e.modelIds.join(", ")}</div></>), sortVal: (e) => e.createdAt },
    { key: "bench", head: "Benchmark", render: (e) => benchmarkById(e.benchmarkId).name },
    { key: "status", head: "Status", render: (e) => (
      <span className="badge"><span className={`dot ${e.status === "complete" ? "ok" : e.status === "failed" ? "bad" : "warn pulse"}`} />{e.status}{e.status !== "complete" && e.status !== "failed" ? ` ${e.progress}%` : ""}</span>
    ) },
    { key: "evals", head: "Evals", num: true, render: (e) => <span className="num">{e.estimates.evals.toLocaleString()}</span>, sortVal: (e) => e.estimates.evals },
    { key: "created", head: "Created", render: (e) => <span className="num muted">{e.createdAt.slice(0, 10)}</span>, sortVal: (e) => e.createdAt },
  ];

  return (
    <>
      <h1 className="page-title">Experiments</h1>
      <p className="page-sub">Every run is preserved with its version. Re-runs create a new version that <em>supersedes</em> — never deletes — the old one. <SynthBadge /></p>
      <DemoBanner />
      <div className="toolbar">
        <select className="input" value={only} onChange={(e) => setOnly(e.target.value)}>
          <option value="all">All statuses</option>
          <option value="complete">complete</option>
          <option value="running">running</option>
          <option value="evaluating">evaluating</option>
          <option value="analysing">analysing</option>
          <option value="queued">queued</option>
        </select>
        <span className="spacer" />
        <Link className="btn primary" to="/experiments/new">+ Run experiment</Link>
      </div>
      <DataTable cols={cols} rows={exps} rowKey={(e) => e.id} linkTo={(e) => `/experiments/${e.id}`} />
    </>
  );
}
