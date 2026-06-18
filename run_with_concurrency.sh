#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RESULTS_DIR="results"
LOG_FILE="${RESULTS_DIR}/run_concurrent_$(date +%Y%m%d_%H%M%S).log"
TIMEOUT=1800

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== Concurrent Benchmark Run ==="

# Models (under 1B, matching overnight_run.sh)
MODELS="smollm-135m,smollm-360m,qwen25-05b,smollm2-135m,smollm2-360m,gemma3-270m,qwen3-06b,qwen35-08b,lfm2-350m,lfm2-700m,granite4-350m"

# Phase 1: stereoset + demographic-bias
log "--- Phase 1: stereoset + demographic-bias ---"
uv run python scripts/run_experiments.py \
    --models "$MODELS" \
    --benchmarks "stereoset,demographic-bias" \
    --output-dir "$RESULTS_DIR" \
    --timeout "$TIMEOUT" 2>&1 | tee -a "$LOG_FILE"

# Phase 2: winobias
log "--- Phase 2: winobias ---"
uv run python scripts/run_experiments.py \
    --models "$MODELS" \
    --benchmarks "winobias" \
    --output-dir "$RESULTS_DIR" \
    --timeout "$TIMEOUT" 2>&1 | tee -a "$LOG_FILE"

# Phase 3: CV screening with concurrency
log "--- Phase 3: CV screening (concurrency=4) ---"
IFS=',' read -ra MODEL_ARR <<< "$MODELS"
for name in "${MODEL_ARR[@]}"; do
    log "  CV screening: $name"
    uv run python -m slm_bias_testing.runner \
        "$name" \
        --benchmark cv-screening \
        --output-dir "$RESULTS_DIR" \
        --timeout "$TIMEOUT" \
        --concurrency 4 2>&1 | tee -a "$LOG_FILE"
done

# Phase 4: temporal analysis
log "--- Phase 4: temporal analysis ---"
uv run python -m slm_bias_testing.temporal \
    --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"

log "=== Done ==="
find "$RESULTS_DIR" -name "results.json" | wc -l | xargs -I{} log "Total results: {}"
