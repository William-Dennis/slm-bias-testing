/* SVG charts II: run-history trends (points only — no fitted trend on
   small-n) and labelled scatter plots. */
import type { RunPoint } from "../data/types";
import { fmt } from "./ui";

export function TrendChart({ history, unit, height = 200 }: {
  history: RunPoint[]; unit: string; height?: number;
}) {
  const W = 560;
  const H = height;
  const padL = 46, padB = 30, padT = 16, padR = 14;
  const vs = history.map((h) => h.value);
  const lo = Math.min(...vs);
  const hi = Math.max(...vs);
  const span = hi - lo || 1;
  const loP = lo - span * 0.3;
  const hiP = hi + span * 0.3;
  const X = (i: number) => padL + (i / Math.max(1, history.length - 1)) * (W - padL - padR);
  const Y = (v: number) => H - padB - ((v - loP) / (hiP - loP)) * (H - padB - padT);
  const pts = history.map((h, i) => `${X(i)},${Y(h.value)}`).join(" ");
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img">
      {[0.25, 0.5, 0.75].map((f) => {
        const v = loP + (hiP - loP) * f;
        const y = H - padB - f * (H - padB - padT);
        return (
          <g key={f}>
            <line x1={padL} x2={W - padR} y1={y} y2={y} className="grid-line" />
            <text x={padL - 8} y={y + 4} textAnchor="end" className="tick">{fmt(v, 1)}</text>
          </g>
        );
      })}
      <polyline points={pts} fill="none" stroke="#6ea8fe" strokeWidth={1.5} strokeDasharray="1 0" opacity={0.8} />
      {history.map((h, i) => (
        <g key={h.runId}>
          <circle cx={X(i)} cy={Y(h.value)} r={4} fill="#0f141b" stroke="#6ea8fe" strokeWidth={2}>
            <title>{`${h.runId} · ${h.timestamp.slice(0, 10)} · ${fmt(h.value)} ${unit} · n=${h.n.toLocaleString()}`}</title>
          </circle>
          <text x={X(i)} y={H - 8} textAnchor="middle" className="tick">v{i + 1}</text>
        </g>
      ))}
    </svg>
  );
}

export interface ScatterPt {
  x: number; y: number; label: string; color: string; title: string;
  showLabel?: boolean;
}

export function ScatterPlot({ pts, xLabel, yLabel, height = 260 }: {
  pts: ScatterPt[]; xLabel: string; yLabel: string; height?: number;
}) {
  const W = 560;
  const H = height;
  const padL = 52, padB = 40, padT = 14, padR = 90;
  // Log-spaced x: model scales span 135M–9B, linear spacing hides the fleet.
  const lx = pts.map((p) => Math.log10(Math.max(1, p.x)));
  const ys = pts.map((p) => p.y);
  const x0 = Math.min(...lx);
  const x1 = Math.max(...lx);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const sx = (x1 - x0) || 1;
  const sy = (y1 - y0) || 1;
  const X = (v: number) => padL + ((Math.log10(Math.max(1, v)) - x0) / sx) * (W - padL - padR);
  const Y = (v: number) => H - padB - ((v - y0) / sy) * (H - padB - padT);
  const xticks = [150, 500, 1500, 5000, 9000].filter((t) => t >= Math.min(...pts.map((p) => p.x)) && t <= Math.max(...pts.map((p) => p.x)));
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img">
      {[0, 0.5, 1].map((f) => {
        const y = H - padB - f * (H - padB - padT);
        return <line key={f} x1={padL} x2={W - padR} y1={y} y2={y} className="grid-line" />;
      })}
      {xticks.map((t) => (
        <text key={t} x={X(t)} y={H - padB + 16} textAnchor="middle" className="tick">{t >= 1000 ? `${(t / 1000).toFixed(1)}B` : `${t}M`}</text>
      ))}
      {pts.map((p) => (
        <g key={p.label}>
          <circle cx={X(p.x)} cy={Y(p.y)} r={5} fill={p.color} opacity={0.85}>
            <title>{p.title}</title>
          </circle>
          {p.showLabel === false ? null : (p.showLabel || pts.length <= 8) && (
            <text x={X(p.x) + 8} y={Y(p.y) + 3} className="tick">{p.label}</text>
          )}
        </g>
      ))}
      <text x={(W - padR + padL) / 2} y={H - 6} textAnchor="middle" className="axis-label">{xLabel} (log scale)</text>
      <text x={12} y={padT} className="axis-label">{yLabel}</text>
    </svg>
  );
}

export function Sparkline({ values, width = 120, height = 32 }: {
  values: number[]; width?: number; height?: number;
}) {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values.map((v, i) =>
    `${(i / Math.max(1, values.length - 1)) * width},${height - 4 - ((v - lo) / span) * (height - 8)}`
  ).join(" ");
  return (
    <svg width={width} height={height} role="img">
      <polyline points={pts} fill="none" stroke="#6ea8fe" strokeWidth={1.5} />
    </svg>
  );
}
