#!/usr/bin/env python3
"""Run all 4 benchmarks sequentially for a single model.

Usage:
    uv run python scripts/run_single_model.py smollm-135m --output-dir results/
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("single_model")

# Import registry and benchmark API
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from slm_bias_testing.registry import get_model  # noqa: E402

BENCHMARKS = ["stereoset", "winobias", "demographic-bias", "cv-screening"]


def _check_ollama() -> bool:
    """Quick HTTP check that Ollama is responding."""
    import urllib.error
    import urllib.request

    url = f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _restart_ollama() -> bool:
    """Kill and restart Ollama server."""
    logger.warning("Attempting Ollama restart...")
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    time.sleep(2)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for it to come up
    for _ in range(10):
        time.sleep(1)
        if _check_ollama():
            logger.info("Ollama restarted successfully")
            return True
    logger.error("Ollama failed to restart")
    return False


def _ensure_ollama() -> bool:
    """Ensure Ollama is running, restart if needed."""
    if _check_ollama():
        return True
    return _restart_ollama()


def _pull_model(ollama_tag: str) -> bool:
    """Pull an Ollama model image if not already present."""
    # Check if already present
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # ollama list output: NAME  ID  SIZE  MODIFIED
        # Check first column for exact tag match
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if parts and parts[0] == ollama_tag:
                logger.info("Model %s already present", ollama_tag)
                return True
    except (subprocess.TimeoutExpired, OSError):
        pass

    logger.info("Pulling model %s ...", ollama_tag)
    try:
        result = subprocess.run(
            ["ollama", "pull", ollama_tag],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        logger.error("Timed out pulling model %s after 600s", ollama_tag)
        return False
    if result.returncode != 0:
        logger.error("Failed to pull model %s: %s", ollama_tag, result.stderr)
        return False
    logger.info("Successfully pulled %s", ollama_tag)
    return True


def run_benchmark(
    model_name: str,
    ollama_tag: str,
    benchmark: str,
    output_dir: str,
    max_samples: int | None,
    n_runs: int,
    concurrency: int,
) -> dict:
    """Run a single benchmark. Returns summary dict."""
    results_dir = os.path.join(output_dir, model_name, benchmark)
    results_file = os.path.join(results_dir, "results.json")

    # Skip if already done
    if os.path.exists(results_file):
        logger.info("  %s: results exist, skipping", benchmark)
        with open(results_file) as f:
            return json.load(f)

    os.makedirs(results_dir, exist_ok=True)

    logger.info("  %s: starting...", benchmark)

    if benchmark == "cv-screening":
        from slm_bias_testing.benchmark import run_benchmark as run_cv

        df = run_cv(
            model_name=ollama_tag,
            output_dir=results_dir,
            max_samples=max_samples,
            n_runs=n_runs,
            concurrency=concurrency,
        )
        summary = {
            "model": model_name,
            "ollama_tag": ollama_tag,
            "benchmark": benchmark,
            "n_records": len(df) if df is not None else 0,
            "mean_score": float(df["score"].mean()) if df is not None and not df.empty else None,
            "std_score": float(df["score"].std()) if df is not None and not df.empty else None,
        }
    else:
        from slm_bias_testing.call_api import DEFAULT_NUM_CTX
        from slm_bias_testing.call_api import Model as ApiModel

        model = ApiModel(model_name=ollama_tag, num_ctx=DEFAULT_NUM_CTX)

        if benchmark == "stereoset":
            from slm_bias_testing.benchmarks.stereoset import StereoSetBenchmark

            bm = StereoSetBenchmark()
        elif benchmark == "demographic-bias":
            from slm_bias_testing.benchmarks.demographic_bias import DemographicBiasBenchmark

            bm = DemographicBiasBenchmark()
        elif benchmark == "winobias":
            from slm_bias_testing.benchmarks.winobias import WinoBiasBenchmark

            bm = WinoBiasBenchmark()
        else:
            return {"benchmark": benchmark, "error": f"Unknown benchmark: {benchmark}"}

        results = bm.evaluate(model, max_samples=max_samples, output_dir=results_dir)
        bm.save_results(results, results_dir)

        summary = {
            "model": model_name,
            "ollama_tag": ollama_tag,
            "benchmark": benchmark,
            "n_examples": results.get("n_examples", 0),
            "max_samples": max_samples,
            "timestamp": datetime.now().isoformat(),
        }
        for key in [
            "overall_stereotype_score",
            "overall_bias_score",
            "bias_score",
            "overall_accuracy",
            "pro_accuracy",
            "anti_accuracy",
        ]:
            if key in results:
                summary[key] = results[key]

    # Atomic write
    tmp_path = results_file + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_path, results_file)

    logger.info("  %s: done", benchmark)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run all benchmarks for a single model")
    parser.add_argument("model", help="Registered model name")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=3, help="CV screening repeats")
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Concurrent threads for cv-screening"
    )
    args = parser.parse_args()

    model_name = args.model
    try:
        model_config = get_model(model_name)
    except KeyError:
        logger.error("Unknown model: %s", model_name)
        sys.exit(1)

    ollama_tag = model_config["ollama_tag"]

    # Ensure Ollama is running
    if not _ensure_ollama():
        logger.error("Ollama not available, exiting")
        sys.exit(1)

    # Pull model
    if not _pull_model(ollama_tag):
        logger.error("Failed to pull model %s", model_name)
        sys.exit(1)

    # Run benchmarks
    start = time.time()
    summaries = []
    failed = []

    for benchmark in BENCHMARKS:
        try:
            summary = run_benchmark(
                model_name=model_name,
                ollama_tag=ollama_tag,
                benchmark=benchmark,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                n_runs=args.n_runs,
                concurrency=args.concurrency,
            )
            summaries.append(summary)
            if "error" in summary:
                failed.append(benchmark)
        except Exception as e:
            logger.error("  %s: FAILED — %s", benchmark, e)
            # Try to restart Ollama if it crashed
            if not _ensure_ollama():
                logger.error("Ollama dead, cannot continue")
                failed.append(benchmark)
                break
            failed.append(benchmark)

    elapsed = time.time() - start

    # JSON summary to stdout
    result = {
        "model": model_name,
        "ollama_tag": ollama_tag,
        "benchmarks_run": len(summaries),
        "benchmarks_failed": len(failed),
        "failed": failed,
        "elapsed_seconds": round(elapsed),
        "summaries": summaries,
    }
    print(json.dumps(result, indent=2))

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
