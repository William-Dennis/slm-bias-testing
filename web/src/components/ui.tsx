/* Shared presentational components: badges, stats, tables,
   methodology panels, evidence trails. */
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import type { HeadlineStat, SigState } from "../data/types";

export const fmt = (x: number, d = 1): string =>
  x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

export function fmtCI(lo: number, hi: number, d = 1): string {
  return `[${fmt(lo, d)}, ${fmt(hi, d)}]`;
}

export function SigBadge({ sig }: { sig: SigState }) {
  if (sig === "significant") return <span className="badge sig">statistically significant</span>;
  if (sig === "uncertain") return <span className="badge uncertain">statistically uncertain</span>;
  return <span className="badge">not significant</span>;
}

export function SynthBadge() {
  return <span className="badge synthetic" title="Synthetic demo record">synthetic</span>;
}

export function DemoBanner() {
  return (
    <div className="demo-banner">
      <strong>Demo workspace.</strong>
      <span>All evaluation numbers in this UI are synthetic and illustrate workflows — not real-world benchmark results. Record-level <code>is_synthetic = true</code>.</span>
    </div>
  );
}

export function Crumbs({ items }: { items: { label: string; to?: string }[] }) {
  return (
    <div className="crumbs">
      {items.map((it, i) => (
        <span key={i} style={{ display: "contents" }}>
          {i > 0 ? <span className="sep">/</span> : null}
          {it.to ? <Link to={it.to}>{it.label}</Link> : <span>{it.label}</span>}
        </span>
      ))}
    </div>
  );
}

/** Headline metric with full supporting context (never a bare number). */
export function Headline({ h, small }: { h: HeadlineStat; small?: boolean }) {
  return (
    <div>
      <div className={`stat-value${small ? " small" : ""}`}>
        {h.unit === "%" ? `${fmt(h.value)}%` : `${fmt(h.value)} ${h.unit}`}
      </div>
      <div className="stat-meta">
        <span className="num">95% CI {fmtCI(h.ciLow, h.ciHigh)}</span>
        {" · "}<span className="num">n = {h.n.toLocaleString()}</span>
        {" · "}<span className="num">{h.effectKind} {fmt(h.effect, 2)}</span>
        {h.pHolm !== null ? <>{" · "}<span className="num">p<sub>Holm</sub> = {h.pHolm}</span></> : null}
      </div>
      <div style={{ marginTop: 8 }}><SigBadge sig={h.sig} /></div>
      <p className="chart-note">{h.note}</p>
    </div>
  );
}

export interface Col<T> {
  key: string;
  head: string;
  num?: boolean;
  render: (row: T) => ReactNode;
  sortVal?: (row: T) => number | string;
}

export function DataTable<T>({ cols, rows, rowKey, linkTo }: {
  cols: Col<T>[]; rows: T[]; rowKey: (r: T) => string; linkTo?: (r: T) => string;
}) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const active = cols.find((c) => c.key === sort?.key);
  const sorted = active?.sortVal
    ? [...rows].sort((a, b) => {
        const va = active.sortVal!(a);
        const vb = active.sortVal!(b);
        const cmp = typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb));
        return cmp * (sort?.dir ?? 1);
      })
    : rows;
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            {cols.map((c) => (
              <th
                key={c.key}
                className={`${c.num ? "n" : ""}${c.sortVal ? " sortable" : ""}`}
                onClick={c.sortVal ? () => setSort({ key: c.key, dir: sort?.key === c.key && sort.dir === 1 ? -1 : 1 }) : undefined}
              >
                {c.head}{sort?.key === c.key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => {
            const to = linkTo?.(r);
            const cells = cols.map((c) => (
              <td key={c.key} className={c.num ? "n" : ""}>{c.render(r)}</td>
            ));
            return to ? (
              <tr key={rowKey(r)} className="clickable" onClick={() => { window.location.hash = `#${to}`; }}>
                {cells}
              </tr>
            ) : (
              <tr key={rowKey(r)}>{cells}</tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
export function Methodology({ what, how, stats, limits }: {
  what: string; how: string[]; stats: string; limits: string[];
}) {
  return (
    <div className="meth">
      <h4>Methodology</h4>
      <p><strong>What is measured:</strong> {what}</p>
      <ul>{how.map((h, i) => <li key={i}>{h}</li>)}</ul>
      <p><strong>Statistics:</strong> {stats}</p>
      <p><strong>Limitations:</strong> {limits.join(" ")}</p>
    </div>
  );
}

export function Trail({ steps }: { steps: { label: string; to?: string }[] }) {
  return (
    <div className="trail">
      {steps.map((s, i) => (
        <span key={i} style={{ display: "contents" }}>
          {i > 0 ? <span className="arrow">→</span> : null}
          {s.to ? <Link className="step" to={s.to}>{s.label}</Link> : <span className="step">{s.label}</span>}
        </span>
      ))}
    </div>
  );
}
