#!/usr/bin/env bash
# overnight_benchmark.sh — Run all missing SLM bias benchmarks overnight.
# Resumes where it left off (skips existing results.json).
# Pushes completed results to origin/main when done.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO_DIR/results"
LOG="$RESULTS_DIR/overnight_$(date +%Y%m%d_%H%M%S).log"

cd "$REPO_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ── Ensure Ollama is running ─────────────────────────────────────────
ensure_ollama() {
    if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        log "Ollama not running — starting..."
        ollama serve >> "$LOG" 2>&1 &
        for i in $(seq 1 30); do
            sleep 2
            if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
                log "Ollama ready after $((i*2))s"
                return 0
            fi
        done
        log "ERROR: Ollama failed to start within 60s"
        return 1
    fi
    log "Ollama already running"
}

# ── Health check with restart ────────────────────────────────────────
health_check() {
    if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        log "Ollama unresponsive — restarting..."
        pkill -f "ollama serve" 2>/dev/null || true
        sleep 3
        ensure_ollama
    fi
}

# ── Main ─────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
log "=== Overnight benchmark run starting ==="

ensure_ollama

# All registered models (smollm first — fastest, then by size)
MODELS=(
    "smollm-135m" "smollm-360m" "smollm2-135m" "smollm2-360m"
    "gemma3-270m" "qwen3-06b" "qwen25-05b" "granite4-350m"
    "lfm2-350m" "lfm2-700m" "qwen35-08b" "qwen25-15b"
    "gemma3-1b" "llama32-1b" "tinyllama" "stablelm2-16b"
)

BENCHMARKS=("cv-screening" "stereoset" "demographic-bias" "winobias")
TIMEOUT=21600
MAX_SAMPLES=""  # full run

completed=0
skipped=0
failed=0

for model in "${MODELS[@]}"; do
    for bench in "${BENCHMARKS[@]}"; do
        result_file="$RESULTS_DIR/$model/$bench/results.json"

        if [ -f "$result_file" ]; then
            log "SKIP $model/$bench (already exists)"
            ((skipped++)) || true
            continue
        fi

        log "RUN $model/$bench ..."
        health_check

        # Run the benchmark with a per-model timeout of 1hr (macOS-compatible)
        run_args=("$model" --benchmark "$bench" --output-dir "$RESULTS_DIR" --timeout "$TIMEOUT")
        if [ -n "$MAX_SAMPLES" ]; then
            run_args+=(--max-samples "$MAX_SAMPLES")
        fi
        if perl -e 'alarm 21600; exec @ARGV' uv run python -m slm_bias_testing.runner \
            "${run_args[@]}" \
            >> "$LOG" 2>&1; then
            log "DONE $model/$bench"
            ((completed++)) || true
        else
            log "FAIL $model/$bench (exit $?)"
            ((failed++)) || true
            # Don't let one failure stop the run
        fi
    done
done

log "=== Summary ==="
log "Completed: $completed"
log "Skipped (already done): $skipped"
log "Failed: $failed"
log "=== Done ==="

# ── Commit & push results (to branch, never to main) ─────────────────
if [ "$completed" -gt 0 ]; then
    log "Syncing git state..."
    git fetch origin >> "$LOG" 2>&1

    BRANCH="results/$(date +%Y-%m-%d)"
    if ! git checkout -b "$BRANCH" 2>>"$LOG" && ! git checkout "$BRANCH" 2>>"$LOG"; then
        log "ERROR: Failed to checkout branch $BRANCH"
        exit 1
    fi

    # Stage new results (only JSON, not logs)
    git add "results/*/*/results.json" >> "$LOG" 2>&1

    if ! git diff --cached --quiet; then
        git commit -m "chore: overnight benchmark results $(date +%Y-%m-%d)
Results: $completed new, $skipped skipped, $failed failed" >> "$LOG" 2>&1
        git push origin "$BRANCH" >> "$LOG" 2>&1 || log "WARN: push failed — resolve manually"
        log "Pushed to branch $BRANCH — create a PR to merge."
    else
        log "No new results to commit"
    fi
    git checkout main >> "$LOG" 2>&1
else
    log "Nothing new to push"
fi
