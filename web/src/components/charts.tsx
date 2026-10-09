/* SVG charts I: group dot-and-interval plots, per-column heat table.
   Points with whiskers on a zero-based axis — no bars, so a small gap
   can never look large through a truncated axis. */
import type { GroupStat } from "../data/types";
import { fmt } from "./ui";

const INK = "#dbe2ec";
const TEAL = "#4fd1a5";
const BLUE = "#6ea8fe";

export function BarsWithCI({ groups, unit, height = 240 }: {
  groups: GroupStat[]; unit: string; height?: number;
}) {
  const W = 560;
  const H = height;
  const padL = 48, padB = 52, padT = 18, padR = 12;
  const hi = Math.max(...groups.map((g) => g.ciHigh));
  const hiP = hi * 1.18 || 1;
  const Y = (v: number) => H - padB - (Math.max(0, v) / hiP) * (H - padB - padT);
  const slot = (W - padL - padR) / groups.length;
  const ticks = [0, 0.5, 1].map((f) => hiP * f);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img">
      {ticks.map((v) => (
        <g key={v}>
          <line x1={padL} x2={W - padR} y1={Y(v)} y2={Y(v)} className="grid-line" />
          <text x={padL - 8} y={Y(v) + 4} textAnchor="end" className="tick">{fmt(v, 0)}</text>
        </g>
      ))}
      <line x1={padL} x2={W - padR} y1={Y(0)} y2={Y(0)} stroke={INK} strokeOpacity=".4" />
      {groups.map((g, i) => {
        const cx = padL + slot * i + slot / 2;
        return (
          <g key={g.group}>
            <line x1={cx} x2={cx} y1={Y(g.ciHigh)} y2={Y(g.ciLow)} stroke={TEAL} strokeWidth={2} />
            <line x1={cx - 8} x2={cx + 8} y1={Y(g.ciHigh)} y2={Y(g.ciHigh)} stroke={TEAL} strokeWidth={2} />
            <line x1={cx - 8} x2={cx + 8} y1={Y(g.ciLow)} y2={Y(g.ciLow)} stroke={TEAL} strokeWidth={2} />
            <circle cx={cx} cy={Y(g.mean)} r={5} fill={BLUE} stroke={INK} strokeWidth={1}>
              <title>{`${g.group}: mean ${fmt(g.mean)} (95% CI ${fmt(g.ciLow)}–${fmt(g.ciHigh)}), n=${g.n.toLocaleString()}`}</title>
            </circle>
            <text x={cx} y={Y(g.ciHigh) - 8} textAnchor="middle" className="tick" fill={INK}>{fmt(g.mean)}{unit === "%" ? "%" : ""}</text>
            <text x={cx} y={H - padB + 16} textAnchor="middle" className="tick">{g.group}</text>
            <text x={cx} y={H - padB + 30} textAnchor="middle" className="tick">n={g.n.toLocaleString()}</text>
          </g>
        );
      })}
      <text x={12} y={16} className="axis-label">mean + 95% CI {unit === "%" ? "(%, zero-based)" : `(${unit}, zero-based)`}</text>
    </svg>
  );
}

export interface HeatCell { value: number | null; n: number; sig: string; note: string }

export function HeatTable({ rows, cols, cells, unitNote }: {
  rows: { id: string; label: string }[];
  cols: { id: string; label: string; full?: string }[];
  cells: HeatCell[][];
  unitNote: string;
}) {
  // Per-column min-max scaling: columns have incompatible units.
  // Neutral blue intensity — red is reserved for significance dots only,
  // so a column leader never reads as an alarm on its own.
  const scales = cols.map((_, c) => {
    const vs = cells.map((r) => r[c].value).filter((v): v is number => v !== null);
    return { lo: Math.min(...vs), hi: Math.max(...vs) };
  });
  const alpha = (c: number, v: number | null): number => {
    if (v === null) return 0;
    const { lo, hi } = scales[c];
    if (hi === lo) return 0.5;
    return 0.06 + 0.5 * ((v - lo) / (hi - lo));
  };
  return (
    <div>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr><th>Model</th>{cols.map((c) => <th key={c.id} className="n" title={c.full ?? c.id}>{c.label}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={r.id}>
                <td><span className="cell-main">{r.label}</span></td>
                {cols.map((c, ci) => {
                  const cell = cells[ri][ci];
                  return (
                    <td key={c.id} className="n" title={cell.value === null ? "no completed run" : `${cell.note} · n=${cell.n.toLocaleString()}`}
                      style={cell.value === null ? { background: "rgba(148,163,184,.07)", color: "var(--text-mute)" } : { background: `rgba(110,168,254,${alpha(ci, cell.value).toFixed(2)})` }}>
                      {cell.value === null ? "n/a" : (
                        <span className="num">{fmt(cell.value)}{" "}
                          <span className={`dot ${cell.sig === "significant" ? "bad" : cell.sig === "uncertain" ? "warn" : "ok"}`}
                            style={{ verticalAlign: "middle" }} />
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="chart-note">
        {unitNote} Colour is scaled <em>within each column only</em> — never compare shades across columns.
        Dots: <span className="dot bad" /> significant · <span className="dot warn" /> uncertain · <span className="dot ok" /> not significant. Hover a cell for n and interval.
      </p>
    </div>
  );
}
