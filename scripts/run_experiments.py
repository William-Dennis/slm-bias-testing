#!/usr/bin/env python3
"""Run experiments across multiple models in small batches.

Usage:
    python run_experiments.py --benchmarks cv-screening,stereoset --max-samples 10 --timeout 600
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

from slm_bias_testing.registry import MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Models under 1B params, sorted by release date (oldest first for temporal analysis)
SLM_MODELS = [
    "smollm-135m",  # 2024-07, 135M
    "smollm-360m",  # 2024-07, 360M
    "qwen25-05b",  # 2024-09, 500M
    "smollm2-135m",  # 2024-11, 135M
    "smollm2-360m",  # 2024-11, 360M
    "gemma3-270m",  # 2025-03, 270M
    "qwen3-06b",  # 2025-04, 600M
    "qwen35-08b",  # 2025-05, 800M
    "lfm2-350m",  # 2025-07, 350M
    "lfm2-700m",  # 2025-07, 700M
    "granite4-350m",  # 2025-10, 350M
]


def _check_ollama():
    """Quick HTTP check that Ollama is responding."""
    import urllib.error
    import urllib.request

    url = "http://localhost:11434/api/tags"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            if attempt < 4:
                time.sleep(2)
    logger.error("Ollama not responding after 5 attempts — run 'ollama serve' first")
    return False


def run_benchmarks(models, benchmarks, output_dir, max_samples, timeout):
    """Run benchmarks across models sequentially."""
    results_summary = []

    for model_name in models:
        if model_name not in MODELS:
            logger.warning("Unknown model: %s, skipping", model_name)
            continue

        # Ensure Ollama is alive before each model (restart if crashed)

        for benchmark in benchmarks:
            logger.info("=" * 60)
            logger.info("Running %s / %s", model_name, benchmark)

            results_dir = os.path.join(output_dir, model_name)
            results_file = os.path.join(results_dir, benchmark, "results.json")

            if os.path.exists(results_file):
                logger.info("Results exist, skipping")
                try:
                    with open(results_file) as f:
                        summary = json.load(f)
                    results_summary.append(summary)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Corrupt results file %s: %s", results_file, e)
                continue

            # Check Ollama is alive before running the benchmark
            if not _check_ollama():
                logger.error("Skipping %s / %s — Ollama not available", model_name, benchmark)
                continue

            cmd = [
                sys.executable,
                "-m",
                "slm_bias_testing.runner",
                model_name,
                "--benchmark",
                benchmark,
                "--output-dir",
                output_dir,
                "--timeout",
                str(timeout),
            ]
            if max_samples is not None:
                cmd.extend(["--max-samples", str(max_samples)])

            start = time.time()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                elapsed = time.time() - start

                if result.returncode == 0:
                    logger.info("Completed in %.1fs", elapsed)
                    if results_file and os.path.exists(results_file):
                        try:
                            with open(results_file) as f:
                                summary = json.load(f)
                            results_summary.append(summary)
                        except (json.JSONDecodeError, OSError) as e:
                            logger.warning("Corrupt results file %s: %s", results_file, e)
                else:
                    logger.error("Failed (exit %d): %s", result.returncode, result.stderr[-500:])
            except subprocess.TimeoutExpired:
                logger.error("Timed out after %ds", timeout)

            # Delay between runs to let Ollama stabilise
            time.sleep(5)

    return results_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    parser.add_argument(
        "--benchmarks", default="cv-screening,stereoset", help="Comma-separated benchmark names"
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    models = SLM_MODELS if args.models == "all" else [m.strip() for m in args.models.split(",")]

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    logger.info("Running %d models x %d benchmarks", len(models), len(benchmarks))
    logger.info("Models: %s", ", ".join(models))
    logger.info("Benchmarks: %s", ", ".join(benchmarks))

    summary = run_benchmarks(models, benchmarks, args.output_dir, args.max_samples, args.timeout)

    # Save summary
    summary_file = os.path.join(args.output_dir, "experiment_summary.json")
    with open(summary_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "n_models": len(models),
                "n_benchmarks": len(benchmarks),
                "max_samples": args.max_samples,
                "results": summary,
            },
            f,
            indent=2,
        )

    logger.info("=" * 60)
    logger.info("Experiment complete. Summary: %s", summary_file)
    for r in summary:
        logger.info("  %s / %s: %s", r.get("model"), r.get("benchmark"), r)


if __name__ == "__main__":
    main()
