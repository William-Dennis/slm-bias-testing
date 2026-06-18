from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
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


def run_benchmark_for_model(
    model_name: str,
    benchmark: str,
    base_output_dir: str,
    timeout: int,
    max_samples: int | None = None,
    concurrency: int = 1,
) -> None:
    """Run benchmark(s) for a single model with resume support."""
    model_config = get_model(model_name)
    ollama_tag = model_config["ollama_tag"]

    # Pull model once before running any benchmarks
    if not pull_model(ollama_tag):
        logger.error("Skipping %s due to pull failure", model_name)
        return

    bench_list = get_benchmarks(benchmark)

    for bench in bench_list:
        results_dir = os.path.join(base_output_dir, model_name, bench)
        results_file = os.path.join(results_dir, "results.json")

        if os.path.exists(results_file):
            logger.info("Results already exist for %s/%s — skipping", model_name, bench)
            continue

        os.makedirs(results_dir, exist_ok=True)

        logger.info("Running benchmark %s with model %s ...", bench, model_name)

        if bench == "cv-screening":
            from slm_bias_testing.benchmark import run_benchmark

            df = run_benchmark(
                model_name=ollama_tag,
                output_dir=results_dir,
                max_samples=max_samples,
                concurrency=concurrency,
            )
            summary = {
                "model": model_name,
                "ollama_tag": ollama_tag,
                "benchmark": bench,
                "n_records": len(df) if df is not None else 0,
                "mean_score": float(df["score"].mean())
                if df is not None and not df.empty
                else None,
                "std_score": float(df["score"].std()) if df is not None and not df.empty else None,
            }
        else:
            from slm_bias_testing.call_api import DEFAULT_NUM_CTX
            from slm_bias_testing.call_api import Model as ApiModel

            model = ApiModel(model_name=ollama_tag, num_ctx=DEFAULT_NUM_CTX)

            if bench == "stereoset":
                from slm_bias_testing.benchmarks.stereoset import StereoSetBenchmark

                bm: BaseBenchmark = StereoSetBenchmark()
            elif bench == "crows-pairs":
                from slm_bias_testing.benchmarks.crows_pairs import CrowsPairsBenchmark

                bm = CrowsPairsBenchmark()
            elif bench == "bbq":
                from slm_bias_testing.benchmarks.bbq import BBQBiasBenchmark

                bm = BBQBiasBenchmark()
            elif bench == "demographic-bias":
                from slm_bias_testing.benchmarks.demographic_bias import DemographicBiasBenchmark

                bm = DemographicBiasBenchmark()
            elif bench == "winobias":
                from slm_bias_testing.benchmarks.winobias import WinoBiasBenchmark

                bm = WinoBiasBenchmark()
            else:
                logger.error("Unknown benchmark: %s", bench)
                continue

            results = bm.evaluate(model, max_samples=max_samples, output_dir=results_dir)
            bm.save_results(results, results_dir)

            summary = {
                "model": model_name,
                "ollama_tag": ollama_tag,
                "benchmark": bench,
                "n_examples": results.get("n_examples", 0),
                "max_samples": max_samples,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }
            if "overall_stereotype_score" in results:
                summary["overall_stereotype_score"] = results["overall_stereotype_score"]
            if "overall_bias_score" in results:
                summary["overall_bias_score"] = results["overall_bias_score"]
            if "bias_score" in results:
                summary["bias_score"] = results["bias_score"]
            if "overall_accuracy" in results:
                summary["overall_accuracy"] = results["overall_accuracy"]
            if "pro_accuracy" in results:
                summary["pro_accuracy"] = results["pro_accuracy"]
            if "anti_accuracy" in results:
                summary["anti_accuracy"] = results["anti_accuracy"]

        tmp_path = results_file + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(summary, f, indent=2)
        os.rename(tmp_path, results_file)

        logger.info("Saved results for %s/%s", model_name, bench)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model batch runner")
    parser.add_argument("models", nargs="+", help="Registered model name(s) to run")
    parser.add_argument(
        "--benchmark",
        default="cv-screening",
        choices=BENCHMARK_CHOICES,
        help="Benchmark to run (default: cv-screening)",
    )
    parser.add_argument("--output-dir", default="results", help="Base output directory")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-model timeout in seconds")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples per benchmark (for testing)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent prediction threads for cv-screening (default: 1). "
        "Set OLLAMA_NUM_PARALLEL on the server to match.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    invalid = [m for m in args.models if m not in MODELS]
    if invalid:
        logger.error("Unknown model(s): %s. Available: %s", ", ".join(invalid), ", ".join(MODELS))
        return

    for model_name in args.models:
        run_benchmark_for_model(
            model_name,
            args.benchmark,
            args.output_dir,
            args.timeout,
            args.max_samples,
            args.concurrency,
        )


if __name__ == "__main__":
    main()
