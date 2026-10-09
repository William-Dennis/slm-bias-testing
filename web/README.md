# Bias Benchmarking Lab — Web UI (demo)

Dark-first React + TypeScript + Vite application for exploring LLM bias
evaluations. **All evaluation numbers are synthetic** (`is_synthetic = true`
at record level) and illustrate researcher workflows — they are not
real-world benchmark results.

## Run it

```bash
npm install
npm run dev        # local dev server
npm run build      # type-check + production build (outputs dist/)
npm run preview    # serve the production build
```

## Structure

```
src/
  data/
    types.ts       # typed domain models (no dict-shaped payloads in UI)
    models.ts      # model catalog, transcribed from ../src/slm_bias_testing/registry.py
    benchmarks.ts  # 12 benchmark defs (4 mirror the Python package, 8 synthetic pilots)
    stats.ts       # Wilson / Newcombe / Cohen's h / Holm estimators + seeded PRNG
    results.ts     # synthetic result generation (intervals computed, never hand-typed)
    pairs.ts       # counterfactual pair generation with evaluator records
    workspace.ts   # experiments (append-only), judges, insights, selectors
  components/      # shell, badges, tables, SVG charts, ⌘K palette
  routes/          # Overview, Models, Benchmarks, Experiments, Compare, Tasks, Insights
shot.mjs           # screenshot helper: node shot.mjs [baseUrl] [outDir]
verify.mjs         # interaction checks (runner flow, deep links, ⌘K)
.qa/screenshots/  # visual-QA evidence for the major routes
```

## Research-integrity rules enforced in UI

- Every headline metric shows 95% CI, n, effect size and Holm-adjusted p.
- No cross-benchmark averages; heatmap colour is per-column with dots for evidence state.
- Trend lines are never fitted on small-n histories (points only).
- Judge outputs are labelled evaluator-assigned scores; disagreement is retained.
- No “fair/unbiased/proves” language — observed disparity, estimated effect,
  statistically significant / uncertain.
