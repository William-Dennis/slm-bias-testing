/* Insights: synthetic findings with measured results cleanly separated
   from generated interpretation — every card links to its evidence. */
import { Link } from "react-router-dom";
import { DemoBanner, SynthBadge } from "../components/ui";
import { INSIGHTS } from "../data/workspace";

export function Insights() {
  const measured = INSIGHTS.filter((i) => i.kind === "measured");
  const interp = INSIGHTS.filter((i) => i.kind === "interpretation");
  return (
    <>
      <h1 className="page-title">Insights</h1>
      <p className="page-sub">
        Generated reading of the demo data. <strong>Measured results</strong> restate statistics shown elsewhere;
        {" "}<strong>generated interpretations</strong> are hypotheses — useful, but not evidence. Both link to supporting data. <SynthBadge />
      </p>
      <DemoBanner />
      <h2 className="section-title">Measured results</h2>
      {measured.map((i) => (
        <div key={i.id} className="card mb">
          <div className="insight-kind measured">Measured result</div>
          <div className="mt" style={{ fontWeight: 650, fontSize: 14 }}>{i.title}</div>
          <p className="chart-note" style={{ fontSize: 13 }}>
            {i.body.split("Interpretation:")[0]}
            {i.body.includes("Interpretation:") ? <><br /><span className="insight-kind interp">Interpretation — </span>{i.body.split("Interpretation:")[1]}</> : null}
          </p>
          <div className="row-flex">{i.links.map((l) => <Link key={l.to} className="btn sm" to={l.to}>{l.label}</Link>)}</div>
        </div>
      ))}
      <h2 className="section-title">Generated interpretations</h2>
      {interp.map((i) => (
        <div key={i.id} className="card mb">
          <div className="insight-kind interp">Generated interpretation</div>
          <div className="mt" style={{ fontWeight: 650, fontSize: 14 }}>{i.title}</div>
          <p className="chart-note" style={{ fontSize: 13 }}>{i.body}</p>
          <div className="row-flex">{i.links.map((l) => <Link key={l.to} className="btn sm" to={l.to}>{l.label}</Link>)}</div>
        </div>
      ))}
    </>
  );
}
