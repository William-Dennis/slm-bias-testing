/* Models: searchable, filterable registry with per-benchmark disparity
   columns — counts of significant gaps, never a single fairness score. */
import { useMemo, useState } from "react";
import { BENCHMARKS } from "../data/benchmarks";
import { MODELS, formatParams } from "../data/models";
import { getResult, modelAccuracy, modelCostC, modelLatency } from "../data/results";
import { modelEvalCount } from "../data/workspace";
import { DataTable, DemoBanner, SynthBadge, fmt, type Col } from "../components/ui";

interface Row {
  id: string; family: string; params: number; release: string; api: string;
  sigGaps: number; worst: string; worstVal: number; ciL: number; ciH: number; n: number;
  latency: number; cost: number; evals: number; acc: number;
}

export function Models() {
  const [q, setQ] = useState("");
  const [fam, setFam] = useState("all");
  const [api, setApi] = useState("all");
  const [bench, setBench] = useState("cv-screening");

  const fams = useMemo(() => [...new Set(MODELS.map((m) => m.family))].sort(), []);

  const rows: Row[] = useMemo(() => MODELS
    .filter((m) => (fam === "all" || m.family === fam) && (api === "all" || m.api === api))
    .filter((m) => m.id.includes(q.trim().toLowerCase()))
    .map((m) => {
      const all = BENCHMARKS.map((b) => getResult(m.id, b.id));
      const sig = all.filter((r) => r.headline.sig === "significant").length;
      const focus = getResult(m.id, bench);
      return {
        id: m.id, family: m.family, params: m.params, release: m.releaseDate, api: m.api,
        sigGaps: sig, worst: focus.headline.label, worstVal: focus.headline.value,
        ciL: focus.headline.ciLow, ciH: focus.headline.ciHigh, n: focus.headline.n,
        latency: modelLatency(m.id), cost: modelCostC(m.id), evals: modelEvalCount(m.id), acc: modelAccuracy(m.id),
      };
    }), [q, fam, api, bench]);

  const cols: Col<Row>[] = [
    { key: "id", head: "Model", render: (r) => (<><span className="cell-main">{r.id}</span><div className="cell-sub">{r.family} · {r.release}</div></>), sortVal: (r) => r.id },
    { key: "params", head: "Size", num: true, render: (r) => <span className="num">{formatParams(r.params)}</span>, sortVal: (r) => r.params },
    { key: "api", head: "API", render: (r) => <span className="badge">{r.api}</span> },
    { key: "worst", head: `Gap · ${(BENCHMARKS.find((b) => b.id === bench) ?? BENCHMARKS[0]).name}`, num: true, render: (r) => (<span><span className="num">{fmt(r.worstVal)} <span className="muted">{getResult(r.id, bench).headline.unit}</span></span><div className="cell-sub num">CI [{fmt(r.ciL)}, {fmt(r.ciH)}] · n={r.n.toLocaleString()}</div></span>), sortVal: (r) => r.worstVal },
    { key: "sig", head: "Significant gaps", num: true, render: (r) => <span className="num">{r.sigGaps}/12</span>, sortVal: (r) => r.sigGaps },
    { key: "acc", head: "Task accuracy*", num: true, render: (r) => <span className="num">{fmt(r.acc, 0)}%</span>, sortVal: (r) => r.acc },
    { key: "lat", head: "Latency", num: true, render: (r) => <span className="num">{r.latency} ms</span>, sortVal: (r) => r.latency },
    { key: "cost", head: "Cost /1k", num: true, render: (r) => <span className="num">¢{r.cost.toFixed(1)}</span>, sortVal: (r) => r.cost },
    { key: "evals", head: "Evals", num: true, render: (r) => <span className="num">{r.evals.toLocaleString()}</span>, sortVal: (r) => r.evals },
  ];

  return (
    <>
      <h1 className="page-title">Models</h1>
      <p className="page-sub">20 registered models across 7 families. Fairness is reported per benchmark with intervals — this product has no single fairness score by design.</p>
      <DemoBanner />
      <div className="toolbar">
        <input type="search" className="input" placeholder="Search models…" value={q} onChange={(e) => setQ(e.target.value)} />
        <select className="input" value={fam} onChange={(e) => setFam(e.target.value)}>
          <option value="all">All families</option>
          {fams.map((f) => <option key={f} value={f}>{f}</option>)}
        </select>
        <select className="input" value={api} onChange={(e) => setApi(e.target.value)}>
          <option value="all">chat + systemone</option>
          <option value="chat">chat</option>
          <option value="systemone">systemone</option>
        </select>
        <select className="input" value={bench} onChange={(e) => setBench(e.target.value)}>
          {BENCHMARKS.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}
        </select>
        <span className="muted">{rows.length} models <SynthBadge /></span>
      </div>
      <DataTable cols={cols} rows={rows} rowKey={(r) => r.id} linkTo={(r) => `/models/${r.id}`} />
      <p className="chart-note">Click a row for full intervals, pairwise tables and raw pairs. *Task accuracy is a descriptive mean across task benchmarks — not an estimated effect, no interval attached.</p>
    </>
  );
}
