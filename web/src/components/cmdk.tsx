/* ⌘K command palette: models, benchmarks, experiments, pair flags. */
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BENCHMARKS } from "../data/benchmarks";
import { MODELS } from "../data/models";
import { EXPERIMENTS } from "../data/workspace";

interface Entry { label: string; sub: string; kind: string; to: string }

const INDEX: Entry[] = [
  ...MODELS.map((m): Entry => ({ label: m.id, sub: `${m.family} · ${(m.params / 1e6).toFixed(0)}M`, kind: "model", to: `/models/${m.id}` })),
  ...BENCHMARKS.map((b): Entry => ({ label: b.name, sub: b.category.join(", "), kind: "benchmark", to: `/benchmarks/${b.id}` })),
  ...EXPERIMENTS.map((e): Entry => ({ label: e.name, sub: `${e.status} · v${e.runVersion}`, kind: "experiment", to: `/experiments/${e.id}` })),
  { label: "Compare models", sub: "side-by-side with intervals", kind: "page", to: "/compare" },
  { label: "Task Explorer — judge disagreements", sub: "pairs needing adjudication", kind: "pairs", to: "/tasks?flag=disagreement" },
  { label: "Task Explorer — largest disparities", sub: "top paired differences", kind: "pairs", to: "/tasks?flag=outlier" },
  { label: "Run a new experiment", sub: "wizard", kind: "page", to: "/experiments/new" },
];

export function CmdK({ onClose }: { onClose: () => void }) {
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const nav = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);

  const hits = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const list = needle
      ? INDEX.filter((e) => `${e.label} ${e.sub}`.toLowerCase().includes(needle))
      : INDEX;
    return list.slice(0, 12);
  }, [q]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(hits.length - 1, s + 1)); }
      if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); }
      if (e.key === "Enter" && hits[sel]) { onClose(); nav(hits[sel].to); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [hits, sel, nav, onClose]);

  return (
    <div className="cmdk-overlay" onClick={onClose}>
      <div className="cmdk" onClick={(e) => e.stopPropagation()}>
        <input ref={inputRef} value={q} onChange={(e) => { setQ(e.target.value); setSel(0); }} placeholder="Search models, benchmarks, experiments…" />
        <div className="cmdk-list">
          {hits.map((h, i) => (
            <div key={`${h.kind}-${h.label}`} className={`cmdk-item${i === sel ? " sel" : ""}`}
              onClick={() => { onClose(); nav(h.to); }}>
              <span>{h.label}</span>
              <span className="muted" style={{ fontSize: 12 }}>{h.sub}</span>
              <span className="kind">{h.kind}</span>
            </div>
          ))}
          {hits.length === 0 ? <div className="empty">No matches.</div> : null}
        </div>
      </div>
    </div>
  );
}
