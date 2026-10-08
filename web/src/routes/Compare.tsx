/* Model comparison: 2–5 models on one benchmark with intervals and
   plain-language deltas against the baseline (first selected). */
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { BarsWithCI } from "../components/charts";
import { DataTable, DemoBanner, SigBadge, SynthBadge, fmt, type Col } from "../components/ui";
import { BENCHMARKS } from "../data/benchmarks";
import { MODELS, formatParams, modelById } from "../data/models";
import { getResult, modelAccuracy, modelCostC, modelLatency } from "../data/results";

export function Compare() {
  const [params] = useSearchParams();
  const initial = useMemo(() => {
    const ms = (params.get("models") ?? "nimble,laya-typed-decisions,gemma3-270m").split(",").filter((x) => MODELS.some((m) => m.id === x));
    return ms.length >= 2 ? ms.slice(0, 5) : ["nimble", "laya-typed-decisions", "gemma3-270m"];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const [models, setModels] = useState<string[]>(initial);
  // Follow in-app links (e.g. from Insights) that change ?models=.
  useEffect(() => {
    const ms = (params.get("models") ?? "").split(",").filter((x) => MODELS.some((m) => m.id === x));
    if (ms.length >= 2) setModels(ms.slice(0, 5));
  }, [params]);
  const [bench, setBench] = useState("cv-screening");
  const b = BENCHMARKS.find((x) => x.id === bench) ?? BENCHMARKS[0];
  const base = models[0];

  const toggle = (id: string) =>
    setModels((ms) => (ms.includes(id) ? (ms.length > 2 ? ms.filter((x) => x !== id) : ms) : ms.length >= 5 ? ms : [...ms, id]));

  const res = useMemo(() => models.map((id) => getResult(id, bench)), [models, bench]);
  const baseVal = res[0]?.headline.value ?? 0;
  const baseHw = res[0] ? (res[0].headline.ciHigh - res[0].headline.ciLow) / 2 : 0;
  /** Approximate 95% interval half-width for a difference of two
   *  independent headline estimates (root-sum-square). */
  const deltaHw = (i: number): number => {
    const hw = (res[i].headline.ciHigh - res[i].headline.ciLow) / 2;
    return Math.sqrt(baseHw * baseHw + hw * hw);
  };

  interface Row { k: string; label: string; vals: string[]; }
  const metricRows: Row[] = [
    { k: "gap", label: `Observed gap (${b.metrics[0].unit})`, vals: res.map((r) => `${fmt(r.headline.value)} (CI ${fmt(r.headline.ciLow)}–${fmt(r.headline.ciHigh)}, n=${r.headline.n.toLocaleString()})`) },
    { k: "delta", label: "Δ vs baseline (approx 95% CI)", vals: res.map((r, i) => i === 0 ? "baseline" : `${r.headline.value - baseVal > 0 ? "+" : ""}${fmt(r.headline.value - baseVal)} ± ${fmt(deltaHw(i))}`) },
    { k: "ev", label: "Evidence", vals: res.map((r) => r.headline.sig) },
    { k: "acc", label: "Task accuracy", vals: models.map((id) => `${fmt(modelAccuracy(id), 0)}%`) },
    { k: "lat", label: "Latency", vals: models.map((id) => `${modelLatency(id)} ms`) },
    { k: "cost", label: "Cost /1k evals", vals: models.map((id) => `¢${modelCostC(id).toFixed(1)}`) },
    { k: "size", label: "Size", vals: models.map((id) => formatParams(modelById(id).params)) },
  ];

  const cols: Col<Row>[] = [
    { key: "m", head: "Metric", render: (r) => <strong>{r.label}</strong> },
    ...models.map((id, i): Col<Row> => ({
      key: id, head: `${id}${i === 0 ? " (baseline)" : ""}`, num: true,
      render: (r) => (r.k === "ev" ? <SigBadge sig={r.vals[i] as "significant" | "not-significant" | "uncertain"} /> : <span className="num">{r.vals[i]}</span>),
    })),
  ];

  return (
    <>
      <h1 className="page-title">Compare</h1>
      <p className="page-sub">Side-by-side observed disparities with intervals. Deltas are descriptive differences vs the baseline — not verdicts. <SynthBadge /></p>
      <DemoBanner />
      <div className="card mb">
        <div className="toolbar">
          <label className="f-label" htmlFor="cb">Benchmark</label>
          <select id="cb" className="input" value={bench} onChange={(e) => setBench(e.target.value)}>
            {BENCHMARKS.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
          </select>
          <span className="hint">{b.researchQuestion}</span>
        </div>
        <div className="row-flex">
          {MODELS.map((m) => (
            <label key={m.id} className="checkline" style={{ marginBottom: 4 }}>
              <input type="checkbox" checked={models.includes(m.id)} onChange={() => toggle(m.id)} />
              <span className="mono" style={{ fontSize: 12 }}>{m.id}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: `repeat(${models.length}, minmax(0,1fr))` }}>
        {res.map((r, i) => {
          const delta = r.headline.value - baseVal;
          return (
            <div key={r.modelId} className="card stat-card">
              <div className="stat-label"><Link to={`/models/${r.modelId}`}>{r.modelId}</Link></div>
              <div className="stat-value">{fmt(r.headline.value)} <span style={{ fontSize: 14 }}>{r.headline.unit}</span></div>
              <div className="stat-meta">CI [{fmt(r.headline.ciLow)}, {fmt(r.headline.ciHigh)}] · n={r.headline.n.toLocaleString()}<br />
                {r.modelId === base ? "baseline" : <>Δ {delta > 0 ? "+" : ""}{fmt(delta)} ± {fmt(deltaHw(i))} {r.headline.unit} vs {base} (approx)</>}
              </div>
              <div style={{ marginTop: 8 }}><SigBadge sig={r.headline.sig} /></div>
            </div>
          );
        })}
      </div>

      <h2 className="section-title">Metric table</h2>
      <DataTable cols={cols} rows={metricRows} rowKey={(r) => r.k} />

      <div className="grid c2 mt">
        {res.slice(0, 2).map((r) => (
          <div key={r.modelId} className="card">
            <p className="chart-q">Group means — <Link to={`/models/${r.modelId}`}>{r.modelId}</Link></p>
            <BarsWithCI groups={r.groups} unit={r.headline.unit} height={210} />
          </div>
        ))}
      </div>
    </>
  );
}
