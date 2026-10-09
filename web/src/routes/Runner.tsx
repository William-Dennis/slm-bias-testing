/* Experiment Runner wizard: models → benchmark → configure →
   review → run (mocked progression) → results. */
import { useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Crumbs, DemoBanner, SynthBadge } from "../components/ui";
import { BENCHMARKS, benchmarkById } from "../data/benchmarks";
import { MODELS, modelById } from "../data/models";
import { JUDGES, saveUserExperiment } from "../data/workspace";
import type { Experiment } from "../data/types";

const STEPS = ["Models", "Benchmark", "Configure", "Review", "Run"];

export function Runner() {
  const [step, setStep] = useState(0);
  const [models, setModels] = useState<string[]>(["smollm2-135m", "gemma3-270m"]);
  const [bench, setBench] = useState("cv-screening");
  const [trials, setTrials] = useState(3);
  const [temp, setTemp] = useState(0.7);
  const [sysPrompt, setSysPrompt] = useState("assistant-default");
  const [judge, setJudge] = useState(JUDGES[0].id);
  const [evalMethod, setEvalMethod] = useState("dual-judge");
  const [randomize, setRandomize] = useState(true);
  const [seed, setSeed] = useState(42);
  const [phase, setPhase] = useState(0);
  const [log, setLog] = useState<string[]>([]);
  const timer = useRef<number | null>(null);
  const nav = useNavigate();

  const b = benchmarkById(bench);
  const est = useMemo(() => {
    const evals = models.length * Math.min(b.tasks, trials * 60);
    const runtimeMin = Math.max(1, Math.round(evals / 55));
    const avgC = models.reduce((a, id) => a + (modelById(id).api === "systemone" ? 0.3 : 0.6), 0) / Math.max(1, models.length);
    return { evals, runtimeMin, costUsd: Math.round(evals * 0.0004 * avgC * 100) / 100 };
  }, [models, trials, b.tasks]);

  const toggle = (id: string) =>
    setModels((ms) => (ms.includes(id) ? ms.filter((x) => x !== id) : ms.length >= 5 ? ms : [...ms, id]));

  const start = () => {
    setStep(4);
    setPhase(0);
    setLog([stamp("queued — run registered (synthetic demo, no real inference)")]);
    const msgs = [
      "queued — run registered (synthetic demo, no real inference)",
      `running — dispatching ${est.evals} evaluations across ${models.length} models`,
      "evaluating — judges grading pairs (dual-judge, disagreement retained)",
      "analysing — intervals, Holm correction, attrition audit",
      "complete — results persisted as a new version",
    ];
    let i = 0;
    timer.current = window.setInterval(() => {
      i += 1;
      setPhase(i);
      setLog((l) => [...l, stamp(msgs[i])]);
      if (i >= 4 && timer.current) {
        window.clearInterval(timer.current);
        const exp: Experiment = {
          id: `exp-user-${Date.now().toString(36)}`,
          name: `User run — ${b.name} (${models.length} models)`,
          benchmarkId: bench, modelIds: [...models], status: "complete", progress: 100,
          config: { trials, temperature: temp, systemPrompt: sysPrompt, judgeId: judge, evalMethod, randomization: randomize, seed },
          estimates: { runtimeMin: est.runtimeMin, costUsd: est.costUsd, evals: est.evals },
          createdAt: new Date().toISOString(), completedAt: new Date().toISOString(),
          isSynthetic: true, runVersion: 1, supersedes: null,
          log: msgs,
        };
        saveUserExperiment(exp);
        window.setTimeout(() => nav(`/experiments/${exp.id}`), 900);
      }
    }, 1100);
  };

  return (
    <>
      <Crumbs items={[{ label: "Experiments", to: "/experiments" }, { label: "New run" }]} />
      <h1 className="page-title">Run experiment <SynthBadge /></h1>
      <p className="page-sub">Configure a bias evaluation. Execution is mocked in this demo — the workflow, estimates and resulting artefacts behave like the real runner.</p>
      <DemoBanner />
      <div className="steps">
        {STEPS.map((s, i) => (
          <div key={s} className={`step-pill${i < step ? " done" : i === step ? " now" : ""}`}>{i + 1}. {s}</div>
        ))}
      </div>
      {step === 0 && (
        <div className="card">
          <p className="chart-q">Select 1–5 models</p>
          <p className="chart-sub">Comparison is capped at five so intervals stay readable.</p>
          {MODELS.map((m) => (
            <label key={m.id} className="checkline">
              <input type="checkbox" checked={models.includes(m.id)} onChange={() => toggle(m.id)} />
              <span><strong>{m.id}</strong> <span className="muted">{m.family} · {(m.params / 1e6).toFixed(0)}M · {m.api}</span></span>
            </label>
          ))}
          <div className="row-flex mt"><span className="spacer" /><button className="btn primary" disabled={models.length === 0} onClick={() => setStep(1)}>Continue →</button></div>
        </div>
      )}

      {step === 1 && (
        <div className="card">
          <p className="chart-q">Select benchmark</p>
          {BENCHMARKS.map((x) => (
            <label key={x.id} className="checkline">
              <input type="radio" name="bench" checked={bench === x.id} onChange={() => setBench(x.id)} />
              <span><strong>{x.name}</strong> {x.isSynthetic ? <span className="badge synthetic">synthetic</span> : <span className="badge live">instrumented</span>}<br /><span className="muted">{x.researchQuestion}</span></span>
            </label>
          ))}
          <div className="row-flex mt"><button className="btn" onClick={() => setStep(0)}>← Back</button><span className="spacer" /><button className="btn primary" onClick={() => setStep(2)}>Continue →</button></div>
        </div>
      )}

      {step === 2 && (
        <div className="card">
          <p className="chart-q">Configure — {b.name}</p>
          <div className="grid c2">
            <div className="field"><label className="f-label">Trials per item</label><input className="input" type="number" min={1} max={10} value={trials} onChange={(e) => setTrials(Number(e.target.value))} /><div className="hint">Repeated runs collapse to per-item means before inference.</div></div>
            <div className="field"><label className="f-label">Temperature</label><input className="input" type="number" min={0} max={2} step={0.1} value={temp} onChange={(e) => setTemp(Number(e.target.value))} /><div className="hint">Recorded in provenance; 0 for rating instruments.</div></div>
            <div className="field"><label className="f-label">System prompt</label><select className="input" value={sysPrompt} onChange={(e) => setSysPrompt(e.target.value)}><option value="assistant-default">assistant-default</option><option value="recruiter-default">recruiter-default</option><option value="rating-default">rating-default</option></select></div>
            <div className="field"><label className="f-label">Judge</label><select className="input" value={judge} onChange={(e) => setJudge(e.target.value)}>{JUDGES.map((j) => <option key={j.id} value={j.id}>{j.model} {j.version} — {j.rubric.split(":")[0]}</option>)}</select><div className="hint">Judges are evaluators, never ground truth. Disagreement is retained.</div></div>
            <div className="field"><label className="f-label">Evaluation method</label><select className="input" value={evalMethod} onChange={(e) => setEvalMethod(e.target.value)}><option value="dual-judge">dual-judge</option><option value="paired-rating">paired-rating</option><option value="regex-parse">regex-parse</option><option value="reference-grade">reference-grade</option></select></div>
            <div className="field"><label className="f-label">Seed</label><input className="input" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} /><div className="hint">Controls presentation order when randomization is on.</div></div>
          </div>
          <label className="checkline"><input type="checkbox" checked={randomize} onChange={(e) => setRandomize(e.target.checked)} /><span><strong>Randomize condition ordering</strong><br /><span className="muted">Counterfactual pairs are shown AB/BA at random; ordering is preserved per pair.</span></span></label>
          <div className="row-flex mt"><button className="btn" onClick={() => setStep(1)}>← Back</button><span className="spacer" /><button className="btn primary" onClick={() => setStep(3)}>Review →</button></div>
        </div>
      )}

      {step === 3 && (
        <div className="card">
          <p className="chart-q">Review configuration</p>
          <dl className="kv">
            <dt>Models</dt><dd>{models.join(", ")}</dd>
            <dt>Benchmark</dt><dd>{b.name} ({b.version})</dd>
            <dt>Trials · temp · seed</dt><dd className="mono">{trials} · {temp} · {seed}{randomize ? " · randomized" : " · fixed order"}</dd>
            <dt>Judge / method</dt><dd>{JUDGES.find((j) => j.id === judge)?.model} / {evalMethod}</dd>
            <dt>Estimated evals</dt><dd className="mono">{est.evals.toLocaleString()}</dd>
            <dt>Estimated runtime</dt><dd className="mono">≈ {est.runtimeMin} min</dd>
            <dt>Estimated cost</dt><dd className="mono">≈ ${est.costUsd.toFixed(2)}</dd>
          </dl>
          <div className="row-flex mt"><button className="btn" onClick={() => setStep(2)}>← Back</button><span className="spacer" /><button className="btn primary" onClick={start}>Run experiment</button></div>
        </div>
      )}

      {step === 4 && (
        <div className="card">
          <p className="chart-q">Running — {["queued", "running", "evaluating", "analysing", "complete"][phase]}</p>
          <div className="progress-track mb"><div className="progress-fill" style={{ width: `${[4, 30, 58, 82, 100][phase]}%` }} /></div>
          <div className="log-box">{log.map((l, i) => <div key={i}>{l}</div>)}</div>
          {phase >= 4 ? <p className="chart-note">Complete — persisting as a new version and opening results… (<Link to="/experiments">all experiments</Link>)</p> : null}
        </div>
      )}
    </>
  );
}

function stamp(msg: string): string {
  return `${new Date().toISOString().slice(11, 19)}  ${msg}`;
}
