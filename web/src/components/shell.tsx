/* Application shell: sidebar navigation + top bar. */
import { useEffect, useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { CmdK } from "./cmdk";

const NAV = [
  { section: "Evaluate" },
  { to: "/", label: "Overview", end: true },
  { to: "/models", label: "Models" },
  { to: "/benchmarks", label: "Benchmarks" },
  { to: "/experiments", label: "Experiments" },
  { to: "/compare", label: "Compare" },
  { section: "Evidence" },
  { to: "/tasks", label: "Task Explorer" },
  { to: "/insights", label: "Insights" },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const nav = useNavigate();
  useCmdK(setOpen);
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">Δ</div>
          <div>
            <div className="brand-name">Bias Lab</div>
            <div className="brand-sub">LLM fairness evaluation</div>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((item, i) =>
            "section" in item ? (
              <div key={i} className="nav-label">{item.section}</div>
            ) : (
              <NavLink key={item.to} to={item.to} end={item.end} className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
                <span className="lbl">{item.label}</span>
              </NavLink>
            )
          )}
        </nav>
        <div style={{ marginTop: "auto", padding: 12 }}>
          <button className="btn primary" style={{ width: "100%", justifyContent: "center" }} onClick={() => nav("/experiments/new")}>
            + Run experiment
          </button>
          <div className="hint" style={{ marginTop: 8, textAlign: "center" }}>demo data · synthetic</div>
        </div>
      </aside>
      <div className="main">
        <div className="topbar">
          <button className="search-trigger" onClick={() => setOpen(true)}>
            <span>Search models, benchmarks, pairs…</span>
            <span className="spacer" />
            <span className="mono muted" style={{ fontSize: 11 }}>⌘K</span>
          </button>
          <div className="topbar-right">
            <span className="badge live"><span className="dot ok" /> demo workspace</span>
            <Link className="btn sm" to="/experiments">Experiments</Link>
          </div>
        </div>
        <main className="page">{children}</main>
      </div>
      {open ? <CmdK onClose={() => setOpen(false)} /> : null}
    </div>
  );
}

function useCmdK(setOpen: (v: boolean) => void) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen(true);
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [setOpen]);
}
