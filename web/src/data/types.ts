/* Typed domain models (part 1: registry + benchmark + stats). */

export interface ModelMeta {
  id: string;
  ollamaTag: string;
  params: number;
  releaseDate: string; // "YYYY-MM"
  family: string;
  architecture: string;
  api: "chat" | "systemone";
}

export interface MetricDef {
  key: string;
  label: string;
  unit: string;
  directionNote: string; // research-safe description, never "less biased"
}

export interface BenchmarkDef {
  id: string;
  name: string;
  category: string[];
  description: string;
  researchQuestion: string;
  methodology: string[];
  controlledVars: string[];
  metrics: MetricDef[];
  limitations: string[];
  tasks: number;
  variablesTested: string[];
  isSynthetic: boolean;
  version: string;
}

export interface GroupStat {
  group: string;
  n: number;
  mean: number;
  sd: number;
  ciLow: number;
  ciHigh: number;
}

export interface PairwiseStat {
  g1: string;
  g2: string;
  effect: number;
  effectKind: "Cohen's d" | "Cohen's h";
  ciLow: number;
  ciHigh: number;
  pHolm: number | null;
  significant: boolean;
}

export type SigState = "significant" | "not-significant" | "uncertain";

export interface HeadlineStat {
  label: string;
  value: number;
  unit: string;
  ciLow: number;
  ciHigh: number;
  n: number;
  effect: number;
  effectKind: "Cohen's d" | "Cohen's h";
  pHolm: number | null;
  sig: SigState;
  note: string;
}

export interface RunPoint {
  runId: string;
  timestamp: string;
  value: number;
  n: number;
}

export interface Attrition {
  planned: number;
  scored: number;
  outstanding: number;
  parseFailures: number;
  apiErrors: number;
}

export interface Provenance {
  benchmarkVersion: string;
  promptSha: string;
  temperature: number;
  nRuns: number;
  seed: number;
  isSynthetic: boolean;
  timestamp: string;
}

export interface ModelBenchmarkResult {
  modelId: string;
  benchmarkId: string;
  headline: HeadlineStat;
  contextMetric: { label: string; value: number; unit: string };
  groups: GroupStat[];
  pairwise: PairwiseStat[];
  attrition: Attrition;
  provenance: Provenance;
  history: RunPoint[];
}
export interface JudgeEval {
  judgeModel: string;
  judgeVersion: string;
  rubric: string;
  temperature: number;
  seed: number;
  orderingShown: "AB" | "BA";
  scoreA: number;
  scoreB: number;
  confidence: number; // 0..1
  rationale: string;
}

export interface CounterfactualPair {
  id: string;
  experimentId: string;
  modelId: string;
  benchmarkId: string;
  task: string;
  changedVariable: string;
  baselineValue: string;
  counterfactualValue: string;
  ordering: "AB" | "BA";
  randomized: boolean;
  seed: number;
  promptA: string;
  responseA: string;
  scoreA: number | null;
  promptB: string;
  responseB: string;
  scoreB: number | null;
  scoreDiff: number | null;
  evaluator: JudgeEval | null;
  secondEvaluator: JudgeEval | null;
  judgeDisagreement: boolean;
  refusal: boolean;
  failed: boolean;
  outlier: boolean;
  stereotypeExample: boolean;
}

export interface JudgeDef {
  id: string;
  provider: string;
  model: string;
  version: string;
  rubric: string;
  temperature: number;
  seed: number;
  promptTemplate: string;
}

export type ExperimentStatus =
  | "queued"
  | "running"
  | "evaluating"
  | "analysing"
  | "complete"
  | "failed";

export interface ExperimentConfig {
  trials: number;
  temperature: number;
  systemPrompt: string;
  judgeId: string;
  evalMethod: string;
  randomization: boolean;
  seed: number;
}

export interface Experiment {
  id: string;
  name: string;
  benchmarkId: string;
  modelIds: string[];
  status: ExperimentStatus;
  progress: number; // 0..100
  config: ExperimentConfig;
  estimates: { runtimeMin: number; costUsd: number; evals: number };
  createdAt: string;
  completedAt: string | null;
  isSynthetic: boolean;
  runVersion: number;
  supersedes: string | null;
  log: string[];
}

export interface Insight {
  id: string;
  title: string;
  kind: "measured" | "interpretation";
  body: string;
  modelIds: string[];
  benchmarkId: string | null;
  links: { label: string; to: string }[];
}
