#!/usr/bin/env python3
"""Run all models in parallel, each handling all 4 benchmarks.

Usage:
    uv run python scripts/run_parallel.py --concurrency 3 --models all
    uv run python scripts/run_parallel.py --concurrency 2 --models smollm-135m,gemma3-270m
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from slm_bias_testing.registry import MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("parallel")

# The 10 models under 1B for the bias paper
SLM_MODELS = [
    "smollm-135m",
    "smollm-360m",
    "qwen25-05b",
    "smollm2-135m",
    "smollm2-360m",
    "gemma3-270m",
    "qwen3-06b",
    "granite4-350m",
    "lfm2-350m",
    "lfm2-700m",
]

BENCHMARKS = ["stereoset", "winobias", "demographic-bias", "cv-screening"]


def _model_done(model_name: str, output_dir: str) -> bool:
    """Check if all 4 benchmarks have results for this model."""
    for bench in BENCHMARKS:
        results_file = os.path.join(output_dir, model_name, bench, "results.json")
        if not os.path.exists(results_file):
            return False
    return True


def _progress_line(
    idx: int, total: int, model: str, status: str, done: int, elapsed: str = ""
) -> str:
    """Format a progress line."""
    done_str = f"{done}/4" if done >= 0 else ""
    return f"  [{idx}/{total}] {model:<18s} {status:<10s} {done_str:<6s} {elapsed}"


def _parse_summary(stdout: str) -> dict | None:
    """Extract JSON from run_single_model.py stdout."""
    try:
        # Find the JSON block (last { ... } in output)
        lines = stdout.strip().split("\n")
        # Find start of JSON
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break
        if json_start is None:
            return None
        json_text = "\n".join(lines[json_start:])
        return json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel model benchmark runner")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of models to run in parallel (default: 3)",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names, or 'all' for the standard 10",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=3, help="CV screening repeats")
    args = parser.parse_args()

    if args.models == "all":
        models = [m for m in SLM_MODELS if m in MODELS]
    else:
        models = [m.strip() for m in args.models.split(",")]
        invalid = [m for m in models if m not in MODELS]
        if invalid:
            logger.error(
                "Unknown model(s): %s. Available: %s", ", ".join(invalid), ", ".join(MODELS)
            )
            sys.exit(1)

    # Filter already-done models
    pending = [m for m in models if not _model_done(m, args.output_dir)]
    skipped = [m for m in models if m not in pending and _model_done(m, args.output_dir)]

    if skipped:
        logger.info("Already complete (%d): %s", len(skipped), ", ".join(skipped))
    if not pending:
        logger.info("All models complete. Nothing to do.")
        return

    logger.info(
        "Running %d models with concurrency=%d (%d skipped already done)",
        len(pending),
        args.concurrency,
        len(skipped),
    )

    start_time = time.time()
    script_path = os.path.join(os.path.dirname(__file__), "run_single_model.py")
    repo_root = os.path.join(os.path.dirname(__file__), "..")

    # State tracking
    running: dict[str, subprocess.Popen] = {}  # model -> Popen
    done_models: dict[str, dict] = {}  # model -> summary
    failed_models: list[str] = []
    model_queue = list(pending)
    retry_queue: list[str] = []  # models to retry once

    def _spawn(model: str) -> subprocess.Popen:
        """Launch run_single_model.py for a model."""
        cmd = [
            sys.executable,
            script_path,
            model,
            "--output-dir",
            args.output_dir,
            "--n-runs",
            str(args.n_runs),
            "--concurrency",
            str(args.concurrency),  # cv-screening threads per model
        ]
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])

        logger.info("Spawning: %s", model)
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=repo_root,
        )

    def _print_status(title: str | None = None) -> None:
        """Print current state of all models."""
        total = len(models)
        if title:
            print(f"\n{'=' * 60}  {title}  {'=' * 60}", file=sys.stderr)

        for i, m in enumerate(models, 1):
            if m in done_models:
                s = done_models[m]
                elapsed = s.get("elapsed_seconds", 0)
                mins = elapsed // 60
                secs = elapsed % 60
                time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
                benchmarks_done = s.get("benchmarks_run", 0)
                benchmarks_failed = s.get("benchmarks_failed", 0)
                if benchmarks_failed:
                    print(_progress_line(i, total, m, "FAIL", benchmarks_done, time_str))
                else:
                    print(_progress_line(i, total, m, "done", benchmarks_done, time_str))
            elif m in running:
                print(_progress_line(i, total, m, "running", -1))
            elif m in failed_models:
                print(_progress_line(i, total, m, "FAIL", -1))
            elif m in retry_queue:
                print(_progress_line(i, total, m, "retry", -1))
            elif m in model_queue:
                print(_progress_line(i, total, m, "queued", -1))
            elif m in skipped:
                print(_progress_line(i, total, m, "skip", 4, ""))

    # Main loop
    while pending or running:
        # Fill slots up to concurrency
        while len(running) < args.concurrency and model_queue:
            model = model_queue.pop(0)
            running[model] = _spawn(model)

        _print_status()

        # Wait for any to finish (short poll)
        finished = []
        for model, proc in running.items():
            if proc.poll() is not None:
                finished.append(model)

        if not finished:
            time.sleep(2)
            continue

        # Process finished models
        for model in finished:
            proc = running.pop(model)
            stdout, stderr = proc.communicate()
            summary = _parse_summary(stdout)

            if proc.returncode == 0 and summary:
                done_models[model] = summary
                logger.info("  %s: completed", model)
            else:
                # Check if it's a retry candidate
                if model not in retry_queue:
                    retry_queue.append(model)
                    logger.warning("  %s: failed (will retry once)", model)
                    logger.warning("  stderr: %s", stderr[-500:] if stderr else "")
                else:
                    failed_models.append(model)
                    logger.error("  %s: failed again, giving up", model)
                    if summary:
                        done_models[model] = summary  # partial results

        # Re-queue retries
        while retry_queue and len(running) < args.concurrency:
            model = retry_queue.pop(0)
            running[model] = _spawn(model)

    elapsed_total = time.time() - start_time

    # Final summary
    print(f"\n{'=' * 72}", file=sys.stderr)
    print("  FINAL SUMMARY", file=sys.stderr)
    print(f"{'=' * 72}", file=sys.stderr)

    header = (
        f"  {'MODEL':<18s} {'STEREO':>7s} {'WINO':>7s} {'DEMO':>7s} {'CV-SCR':>7s} {'TIME':>8s}"
    )
    print(header, file=sys.stderr)
    print(f"  {'-' * 18} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8}", file=sys.stderr)

    for m in models:
        if m in done_models:
            s = done_models[m]
            elapsed = s.get("elapsed_seconds", 0)
            mins = elapsed // 60
            secs = elapsed % 60
            time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"

            stereo = "--"
            wino = "--"
            demo = "--"
            cvscr = "--"

            for sm in s.get("summaries", []):
                bench = sm.get("benchmark", "")
                if bench == "stereoset":
                    val = sm.get("overall_stereotype_score")
                    stereo = f"{val:.1f}" if val is not None else "OK"
                elif bench == "winobias":
                    val = sm.get("overall_accuracy")
                    wino = f"{val:.1f}" if val is not None else "OK"
                elif bench == "demographic-bias":
                    val = sm.get("bias_score")
                    demo = f"{val:.1f}" if val is not None else "OK"
                elif bench == "cv-screening":
                    val = sm.get("mean_score")
                    cvscr = f"{val:.1f}" if val is not None else "OK"

            fail_tag = "*" if s.get("benchmarks_failed", 0) > 0 else ""
            print(
                f"  {m:<18s} {stereo:>7s} {wino:>7s} {demo:>7s} {cvscr:>7s} {time_str:>8s}{fail_tag}",
                file=sys.stderr,
            )
        elif m in failed_models:
            print(f"  {m:<18s} {'FAIL':>7s}", file=sys.stderr)
        elif m in skipped:
            print(f"  {m:<18s} {'skip':>7s}", file=sys.stderr)

    total_h = int(elapsed_total // 3600)
    total_m = int((elapsed_total % 3600) // 60)
    total_s = int(elapsed_total % 60)
    time_parts = []
    if total_h:
        time_parts.append(f"{total_h}h")
    if total_m:
        time_parts.append(f"{total_m}m")
    time_parts.append(f"{total_s}s")

    print(
        f"\n  Completed: {len(done_models)}/{len(models)}, Failed: {len(failed_models)}",
        file=sys.stderr,
    )
    print(f"  Total: {' '.join(time_parts)}", file=sys.stderr)
    print(f"{'=' * 72}", file=sys.stderr)


if __name__ == "__main__":
    main()
