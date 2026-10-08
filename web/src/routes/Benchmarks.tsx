/* Benchmark library: 12 benchmarks across 14 disparity categories,
   with methodology previews and best-observed models per benchmark. */
import { useMemo, useState } from "react";
import { BENCHMARKS } from "../data/benchmarks";
import { MODELS } from "../data/models";
import { getResult } from "../data/results";
import { experimentsForBenchmark } from "../data/workspace";
import { DataTable, DemoBanner, SynthBadge, fmt, type Col } from "../components/ui";

interface Row {
  id: string; name: string; cats: string; tasks: number; vars: string;
  runs: number; best: string; bestVal: string; synthetic: boolean;
}

const CATS = [...new Set(BENCHMARKS.flatMap((b) => b.category))].sort();

export function Benchmarks() {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("all");

  const rows: Row[] = useMemo(() => BENCHMARKS
    .filter((b) => cat === "all" || b.category.includes(cat))
    .filter((b) => `${b.name} ${b.description}`.toLowerCase().includes(q.trim().toLowerCase()))
    .map((b) => {
      const all = MODELS.map((m) => getResult(m.id, b.id));
      const best = [...all].sort((a, z) => a.headline.value - z.headline.value)[0];
      return {
        id: b.id, name: b.name, cats: b.category.join(", "), tasks: b.tasks,
        vars: b.variablesTested.join("; "), runs: experimentsForBenchmark(b.id).length,
        best: best.modelId, bestVal: `${fmt(best.headline.value)} ${best.headline.unit}`,
        synthetic: b.isSynthetic,
      };
    }), [q, cat]);

  const cols: Col<Row>[] = [
    { key: "name", head: "Benchmark", render: (r) => (<><span className="cell-main">{r.name}</span> <span className="cell-sub">{r.id}</span><div className="cell-sub">{r.cats}</div></>), sortVal: (r) => r.name },
    { key: "tasks", head: "Tasks", num: true, render: (r) => <span className="num">{r.tasks.toLocaleString()}</span>, sortVal: (r) => r.tasks },
    { key: "vars", head: "Variables tested", render: (r) => <span className="dim" style={{ fontSize: 12 }}>{r.vars}</span> },
    { key: "runs", head: "Experiments", num: true, render: (r) => <span className="num">{r.runs}</span>, sortVal: (r) => r.runs },
    { key: "best", head: "Smallest observed gap", render: (r) => (<span><span className="cell-main">{r.best}</span> <span className="num muted">{r.bestVal}</span></span>), sortVal: (r) => r.bestVal },
    { key: "kind", head: "Kind", render: (r) => (r.synthetic ? <SynthBadge /> : <span className="badge live">instrumented</span>) },
  ];

  return (
    <>
      <h1 className="page-title">Benchmarks</h1>
      <p className="page-sub">
        4 instrumented benchmarks mirror the Python package; 8 synthetic pilots validate workflows.
        “Smallest observed gap” is descriptive, not a fairness verdict — open a benchmark for intervals and evidence.
      </p>
      <DemoBanner />
      <div className="toolbar">
        <input type="search" className="input" placeholder="Search benchmarks…" value={q} onChange={(e) => setQ(e.target.value)} />
        <select className="input" value={cat} onChange={(e) => setCat(e.target.value)}>
          <option value="all">All categories</option>
          {CATS.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <span className="muted">{rows.length} benchmarks</span>
      </div>
      <DataTable cols={cols} rows={rows} rowKey={(r) => r.id} linkTo={(r) => `/benchmarks/${r.id}`} />
    </>
  );
}
