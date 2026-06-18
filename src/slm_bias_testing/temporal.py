"""Temporal analysis: bias score vs model release date.

Scans results/ for completed evaluations, merges with model registry
metadata, and produces publication-quality trend plots.

Usage:
    uv run python -m slm_bias_testing.temporal
    uv run python -m slm_bias_testing.temporal --results-dir results/2026-06-10_0200
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from slm_bias_testing.registry import MODELS

logger = logging.getLogger(__name__)

# Score field names used by each benchmark
BENCHMARK_SCORE_FIELDS = {
    "stereoset": "overall_stereotype_score",
    "demographic-bias": None,  # No single score; uses output-length disparity
    "cv-screening": "mean_score",
    "winobias": "bias_score",
}

BENCHMARK_LABELS = {
    "stereoset": "StereoScore (lower = less biased)",
    "demographic-bias": "Output Length Disparity",
    "cv-screening": "Mean CV Score (out of 100)",
    "winobias": "WinoBias Bias Score (pro-anti accuracy gap)",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)


def find_results(base_dir: str = "results") -> list[dict]:
    """Scan results directory and load all result summaries."""
    records = []
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if f == "results.json":
                path = os.path.join(root, f)
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    logger.warning("Skipping corrupted results file: %s", path)
                    continue
                model = data.get("model")
                benchmark = data.get("benchmark")
                if not model or not benchmark:
                    continue
                score_field = BENCHMARK_SCORE_FIELDS.get(benchmark)
                score = data.get(score_field) if score_field else None
                records.append(
                    {
                        "model": model,
                        "benchmark": benchmark,
                        "score": score,
                        "n_examples": data.get("n_examples") or data.get("n_records"),
                        "path": path,
                        "data": data,
                    }
                )
    return records


def merge_registry(records: list[dict]) -> pd.DataFrame:
    """Merge result records with model registry metadata."""
    rows = []
    for r in records:
        name = r["model"]
        if name not in MODELS:
            continue
        cfg: dict[str, Any] = MODELS[name]
        release_date: str = str(cfg["release_date"])
        try:
            release = datetime.strptime(release_date, "%Y-%m-%d")
        except ValueError:
            try:
                release = datetime.strptime(release_date, "%Y-%m")
            except ValueError:
                logger.warning("Unrecognised date format for %s: %s", name, release_date)
                continue
        rows.append(
            {
                "model": name,
                "benchmark": r["benchmark"],
                "score": r["score"],
                "n_examples": r["n_examples"],
                "params": cfg["params"],
                "family": cfg["family"],
                "architecture": cfg.get("architecture", "unknown"),
                "release_date": release_date,
                "release_dt": release,
                "release_ordinal": release.toordinal(),
            }
        )
    return pd.DataFrame(rows)


def plot_temporal(df: pd.DataFrame, output_dir: str = "figs") -> str:
    """Generate publication-quality temporal trend plots.

    Returns path to the output figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    benchmarks = df["benchmark"].unique()
    n_benchmarks = len(benchmarks)
    _fig, axes = plt.subplots(1, n_benchmarks, figsize=(7 * n_benchmarks, 5), squeeze=False)
    axes = axes[0]

    for ax, benchmark in zip(axes, benchmarks, strict=True):
        bdf = df[df["benchmark"] == benchmark].dropna(subset=["score"]).copy()
        if bdf.empty:
            ax.set_title(f"{benchmark}\n(no data)")
            continue

        # Scatter: x = release date, y = bias score
        x = bdf["release_ordinal"].values
        y = bdf["score"].values

        # Color by family
        families = bdf["family"].unique()
        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(families)))
        family_color = {f: c for f, c in zip(families, colors, strict=True)}

        for family in families:
            mask = bdf["family"] == family
            ax.scatter(
                bdf.loc[mask, "release_dt"],
                bdf.loc[mask, "score"],
                label=family,
                color=family_color[family],
                s=80,
                zorder=5,
            )

        # Linear regression trend line
        if len(x) >= 3:
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 3:
                x = x[mask]
                y = y[mask]
                slope, intercept, r_val, p_val, _std_err = sp_stats.linregress(x, y)
            else:
                ax.set_title(BENCHMARK_LABELS.get(benchmark, benchmark), fontsize=10)
                continue
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept

            # Convert x_line back to datetime for plotting
            x_dates = [datetime.fromordinal(int(xi)) for xi in x_line]
            ax.plot(x_dates, y_line, color="red", linewidth=1.5, linestyle="--", alpha=0.7)

            # Confidence band
            y_pred = slope * x + intercept
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (len(x) - 2)
            se_fit = np.sqrt(
                mse * (1 / len(x) + (x_line - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
            )
            t_val = sp_stats.t.ppf(0.975, len(x) - 2)
            ci_upper = y_line + t_val * se_fit
            ci_lower = y_line - t_val * se_fit

            ci_dates = [datetime.fromordinal(int(xi)) for xi in x_line]
            ax.fill_between(ci_dates, ci_lower, ci_upper, alpha=0.15, color="red")

            # Annotation
            direction = "increasing" if slope > 0 else "decreasing"
            sig = "p < 0.05" if p_val < 0.05 else f"p = {p_val:.3f}"
            label = BENCHMARK_LABELS.get(benchmark, benchmark)
            ax.set_title(f"{label}\nR²={r_val**2:.3f}, {sig} ({direction})", fontsize=10)
        else:
            ax.set_title(BENCHMARK_LABELS.get(benchmark, benchmark), fontsize=10)

        ax.set_xlabel("Release Date")
        ax.legend(fontsize=8, loc="best")

    # Tighten layout
    plt.tight_layout()
    path = os.path.join(output_dir, "temporal_trends.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(_fig)
    return path


def plot_family_comparison(df: pd.DataFrame, output_dir: str = "figs") -> str:
    """Plot per-benchmark scores grouped by model family."""
    os.makedirs(output_dir, exist_ok=True)
    benchmarks = df["benchmark"].unique()
    n = len(benchmarks)
    _fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, benchmark in zip(axes, benchmarks, strict=True):
        bdf = df[(df["benchmark"] == benchmark) & df["score"].notna()]
        if bdf.empty:
            ax.set_title(f"{benchmark}\n(no data)")
            continue

        families = bdf.groupby("family")["score"].agg(["mean", "std", "count"])
        families = families.sort_values("mean")

        ax.barh(
            range(len(families)),
            families["mean"],
            xerr=families["std"] / np.sqrt(families["count"].clip(lower=1)),
            tick_label=families.index,
            color="steelblue",
            alpha=0.8,
        )
        ax.set_xlabel(BENCHMARK_LABELS.get(benchmark, benchmark))
        ax.set_title(f"{benchmark} by Family")

    plt.tight_layout()
    path = os.path.join(output_dir, "family_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(_fig)
    return path


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary table."""
    print()
    print("=" * 80)
    print("TEMPORAL ANALYSIS SUMMARY")
    print("=" * 80)

    if df.empty:
        print("No results found. Run benchmarks first.")
        return

    print(f"Models with results: {df['model'].nunique()}")
    print(f"Benchmarks: {', '.join(sorted(df['benchmark'].unique()))}")
    print(f"Date range: {df['release_date'].min()} to {df['release_date'].max()}")
    print(f"Families: {', '.join(sorted(df['family'].unique()))}")
    print()

    for benchmark in sorted(df["benchmark"].unique()):
        bdf = df[df["benchmark"] == benchmark].dropna(subset=["score"])
        print(f"--- {benchmark} ---")
        for _, row in bdf.iterrows():
            print(
                f"  {row['model']:20s} {row['release_date']}  {row['family']:15s} score={row['score']}"
            )
        print()


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Temporal bias analysis")
    parser.add_argument("--results-dir", default="results", help="Results directory to scan")
    parser.add_argument("--output-dir", default="figs", help="Output directory for figures")
    args = parser.parse_args()

    records = find_results(args.results_dir)
    if not records:
        print(f"No results found in {args.results_dir}/")
        print("Run benchmarks first, e.g.:")
        print("  bash scripts/overnight_run.sh")
        sys.exit(0)

    df = merge_registry(records)
    if df.empty:
        print("Results found but none match registered models.")
        sys.exit(0)

    print_summary(df)

    # Generate plots
    temporal_path = plot_temporal(df, args.output_dir)
    print(f"Temporal trend plot: {temporal_path}")

    family_path = plot_family_comparison(df, args.output_dir)
    print(f"Family comparison plot: {family_path}")

    return df


if __name__ == "__main__":
    main()
