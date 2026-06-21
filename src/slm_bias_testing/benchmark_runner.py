"""CLI runner for per-model benchmarks — uses the Node.js pool for parallelism."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING

from slm_bias_testing.registry import MODELS, get_model

if TYPE_CHECKING:
    from slm_bias_testing.benchmarks import BaseBenchmark

logger = logging.getLogger(__name__)

BENCHMARK_CHOICES = ["cv-screening", "stereoset", "demographic-bias", "winobias", "all"]


def get_benchmarks(benchmark: str) -> list[str]:
    if benchmark == "all":
        return ["cv-screening", "stereoset", "demographic-bias", "winobias"]
    return [benchmark]


def pull_model(ollama_tag: str) -> bool:
    """Pull an Ollama model image if not already present."""
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


def run_model_benchmarks(
    model_name: str,
    benchmark: str,
    base_output_dir: str,
    max_samples: int | None = None,
    pool_size: int = 4,
    batch_size: int = 40,
    n_runs: int = 3,
    adaptive: bool = True,
) -> None:
    """Run benchmark(s) for a single model with resume support."""
    from slm_bias_testing.model_clients import OllamaPoolClient

    model_config = get_model(model_name)
    ollama_tag = model_config["ollama_tag"]

    if not pull_model(ollama_tag):
        logger.error("Skipping %s due to pull failure", model_name)
        return

    bench_list = get_benchmarks(benchmark)

    # Create pool client once per model — it manages Ollama lifecycle
    pool_client: OllamaPoolClient | None = None
    try:
        pool_client = OllamaPoolClient(
            model_name=ollama_tag,
            pool_size=pool_size,
            batch_size=batch_size,
            adaptive=adaptive,
        )
        for bench in bench_list:
            results_dir = os.path.join(base_output_dir, model_name, bench)
            results_file = os.path.join(results_dir, "results.json")

            if os.path.exists(results_file):
                logger.info("Results already exist for %s/%s — skipping", model_name, bench)
                continue

            os.makedirs(results_dir, exist_ok=True)
            logger.info("Running benchmark %s with model %s ...", bench, model_name)

            if bench == "cv-screening":
                from slm_bias_testing.cv_screening import run_cv_screening

                df = run_cv_screening(
                    output_dir=results_dir,
                    max_samples=max_samples,
                    pool_client=pool_client,
                    n_runs=n_runs,
                )
                summary = {
                    "model": model_name,
                    "ollama_tag": ollama_tag,
                    "benchmark": bench,
                    "n_records": len(df) if df is not None else 0,
                    "mean_score": float(df["score"].mean())
                    if df is not None and not df.empty
                    else None,
                    "std_score": float(df["score"].std())
                    if df is not None and not df.empty
                    else None,
                }
            else:
                bm = _get_benchmark(bench)
                if bm is None:
                    continue
                results = bm.evaluate(
                    model=None,
                    max_samples=max_samples,
                    output_dir=results_dir,
                    pool_client=pool_client,
                )
                bm.save_results(results, results_dir)
                summary = _build_benchmark_summary(
                    model_name, ollama_tag, bench, results, max_samples
                )

            _write_summary(results_file, summary)
            logger.info("Saved results for %s/%s", model_name, bench)
    finally:
        if pool_client is not None:
            pool_client.close()


def _get_benchmark(bench: str) -> BaseBenchmark | None:
    if bench == "stereoset":
        from slm_bias_testing.benchmarks.stereoset import StereoSetBenchmark

        return StereoSetBenchmark()
    if bench == "demographic-bias":
        from slm_bias_testing.benchmarks.demographic_bias import DemographicBiasBenchmark

        return DemographicBiasBenchmark()
    if bench == "winobias":
        from slm_bias_testing.benchmarks.winobias import WinoBiasBenchmark

        return WinoBiasBenchmark()
    logger.error("Unknown benchmark: %s", bench)
    return None


def _build_benchmark_summary(
    model_name: str,
    ollama_tag: str,
    bench: str,
    results: dict,
    max_samples: int | None,
) -> dict:
    summary: dict = {
        "model": model_name,
        "ollama_tag": ollama_tag,
        "benchmark": bench,
        "n_examples": results.get("n_examples", 0),
        "max_samples": max_samples,
        "timestamp": datetime.now().isoformat(),
    }
    for key in (
        "overall_stereotype_score",
        "overall_bias_score",
        "bias_score",
        "overall_accuracy",
        "pro_accuracy",
        "anti_accuracy",
    ):
        if key in results:
            summary[key] = results[key]
    return summary


def _write_summary(results_file: str, summary: dict) -> None:
    tmp_path = results_file + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_path, results_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model batch runner")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated list of registered model names, or 'all' (default: all)",
    )
    parser.add_argument(
        "--benchmark",
        default="cv-screening",
        choices=BENCHMARK_CHOICES,
        help="Benchmark to run (default: cv-screening)",
    )
    parser.add_argument("--output-dir", default="results", help="Base output directory")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples per benchmark (for testing)"
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=4,
        help="Number of concurrent Ollama workers in the pool (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Number of jobs to send per batch to the pool (default: 40)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of repeated runs per CV in cv-screening (default: 3)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adaptive",
        action="store_true",
        dest="adaptive",
        default=True,
        help="Enable adaptive concurrency in the pool (default)",
    )
    group.add_argument(
        "--no-adaptive",
        action="store_false",
        dest="adaptive",
        help="Disable adaptive concurrency in the pool",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.models == "all":
        model_names = list(MODELS)
    else:
        model_names = [m.strip() for m in args.models.split(",")]

    invalid = [m for m in model_names if m not in MODELS]
    if invalid:
        logger.error(
            "Unknown model(s): %s. Available: %s",
            ", ".join(invalid),
            ", ".join(MODELS),
        )
        return

    for model_name in model_names:
        run_model_benchmarks(
            model_name,
            args.benchmark,
            args.output_dir,
            args.max_samples,
            pool_size=args.pool_size,
            batch_size=args.batch_size,
            n_runs=args.n_runs,
            adaptive=args.adaptive,
        )


if __name__ == "__main__":
    main()
