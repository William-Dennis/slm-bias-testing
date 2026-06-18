"""Cross-model benchmark visualisations.

Scans results/ for all completed evaluations and produces
publication-quality comparison plots.

Usage:
    uv run python -m slm_bias_testing.visualisations
    uv run python -m slm_bias_testing.visualisations --results-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from slm_bias_testing.registry import MODELS

logger = logging.getLogger(__name__)

# ── Score extraction per benchmark ──────────────────────────────────────────

BENCHMARK_SCORE_FIELDS: dict[str, str | None] = {
    "stereoset": "overall_stereotype_score",
    "winobias": "bias_score",
    "demographic-bias": None,  # computed from per_group
    "cv-screening": "mean_score",
}

BENCHMARK_LABELS: dict[str, str] = {
    "stereoset": "StereoScore\n(lower = less biased)",
    "winobias": "WinoBias Bias Score\n(0 = no gender bias)",
    "demographic-bias": "Output Length Disparity %\n(lower = less biased)",
    "cv-screening": "Mean CV Score\n(out of 100)",
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


# ── Data loading ────────────────────────────────────────────────────────────


def _load_benchmark_data(path: str, benchmark: str) -> dict[str, Any] | None:
    """Load a single benchmark result file."""
    try:
        with open(path) as fh:
            return json.load(fh)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        logger.warning("Skipping corrupted results file: %s", path)
        return None


def _extract_score(data: dict[str, Any], benchmark: str) -> float | None:
    """Extract a single numeric score from benchmark results."""
    if benchmark == "demographic-bias":
        pg = data.get("per_group", {})
        male_len = float(pg.get("gender_male", {}).get("avg_output_length", 0))
        female_len = float(pg.get("gender_female", {}).get("avg_output_length", 0))
        if male_len and female_len:
            return round(abs(male_len - female_len) / max(male_len, female_len) * 100, 2)
        return None
    field = BENCHMARK_SCORE_FIELDS.get(benchmark)
    if field and field in data:
        val = data[field]
        return float(val)
    return None


def load_all_results(base_dir: str = "results") -> pd.DataFrame:
    """Scan results directory and build a DataFrame of all model x benchmark scores."""
    rows: list[dict[str, Any]] = []
    benchmark_files = {
        "stereoset": "stereoset.json",
        "winobias": "winobias.json",
        "demographic-bias": "demographic-bias.json",
        "cv-screening": "cv-screening.json",
    }

    for model_dir in sorted(os.listdir(base_dir)):
        model_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_path) or model_dir.startswith("."):
            continue
        for benchmark, filename in benchmark_files.items():
            result_path = os.path.join(model_path, benchmark, filename)
            if not os.path.isfile(result_path):
                continue
            data = _load_benchmark_data(result_path, benchmark)
            if data is None:
                continue
            score = _extract_score(data, benchmark)
            meta = MODELS.get(model_dir, {})
            rows.append(
                {
                    "model": model_dir,
                    "benchmark": benchmark,
                    "score": score,
                    "params": meta.get("params"),
                    "family": meta.get("family", "unknown"),
                    "release_date": meta.get("release_date", ""),
                    "n_examples": data.get("n_examples"),
                }
            )

    return pd.DataFrame(rows)


def load_per_group_data(
    base_dir: str = "results", benchmark: str = "demographic-bias"
) -> pd.DataFrame:
    """Load per-group breakdown for demographic-bias benchmark."""
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(os.listdir(base_dir)):
        result_path = os.path.join(base_dir, model_dir, benchmark, f"{benchmark}.json")
        if not os.path.isfile(result_path):
            continue
        data = _load_benchmark_data(result_path, benchmark)
        if data is None:
            continue
        for group, info in data.get("per_group", {}).items():
            rows.append(
                {
                    "model": model_dir,
                    "group": group,
                    "n": info.get("n", 0),
                    "avg_output_length": info.get("avg_output_length", 0),
                }
            )
    return pd.DataFrame(rows)


def load_per_category_data(base_dir: str = "results", benchmark: str = "stereoset") -> pd.DataFrame:
    """Load per-category breakdown for stereoset benchmark."""
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(os.listdir(base_dir)):
        result_path = os.path.join(base_dir, model_dir, benchmark, f"{benchmark}.json")
        if not os.path.isfile(result_path):
            continue
        data = _load_benchmark_data(result_path, benchmark)
        if data is None:
            continue
        for category, score in data.get("per_category", {}).items():
            rows.append({"model": model_dir, "category": category, "score": score})
    return pd.DataFrame(rows)


def load_per_pronoun_data(base_dir: str = "results", benchmark: str = "winobias") -> pd.DataFrame:
    """Load per-pronoun breakdown for winobias benchmark."""
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(os.listdir(base_dir)):
        result_path = os.path.join(base_dir, model_dir, benchmark, f"{benchmark}.json")
        if not os.path.isfile(result_path):
            continue
        data = _load_benchmark_data(result_path, benchmark)
        if data is None:
            continue
        for pronoun, accuracy in data.get("per_pronoun", {}).items():
            rows.append({"model": model_dir, "pronoun": pronoun, "accuracy": accuracy})
    return pd.DataFrame(rows)


# ── Plot 1: Cross-model heatmap ─────────────────────────────────────────────


def plot_heatmap(df: pd.DataFrame, output_dir: str = "figs") -> str:
    """Heatmap of all models x all benchmarks."""
    os.makedirs(output_dir, exist_ok=True)

    pivot = df.pivot_table(index="model", columns="benchmark", values="score")
    if pivot.empty:
        logger.warning("No data for heatmap")
        return ""

    # Sort by mean score across benchmarks (lower = better for most)
    mean_scores = pivot.mean(axis=1)
    if isinstance(mean_scores, pd.Series):
        pivot = pivot.loc[mean_scores.sort_values().index]

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Bias Score"},
    )
    ax.set_title("Cross-Model Benchmark Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(output_dir, "cross_model_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 2: StereoSet category breakdown ────────────────────────────────────


def plot_stereoset_categories(df: pd.DataFrame, output_dir: str = "figs") -> str:
    """Grouped bar chart of StereoSet scores by category across models."""
    os.makedirs(output_dir, exist_ok=True)

    if df.empty:
        logger.warning("No data for StereoSet category plot")
        return ""

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="model", y="score", hue="category", ax=ax)
    ax.set_title("StereoSet Scores by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Stereotype Score (lower = less biased)")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, "stereoset_categories.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 3: WinoBias pronoun accuracy ───────────────────────────────────────


def plot_winobias_pronouns(df: pd.DataFrame, output_dir: str = "figs") -> str:
    """Grouped bar chart of WinoBias accuracy by pronoun across models."""
    os.makedirs(output_dir, exist_ok=True)

    if df.empty:
        logger.warning("No data for WinoBias pronoun plot")
        return ""

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="model", y="accuracy", hue="pronoun", ax=ax)
    ax.set_title("WinoBias Accuracy by Pronoun", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(title="Pronoun", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, "winobias_pronouns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 4: Demographic bias group comparison ───────────────────────────────


def plot_demographic_groups(df: pd.DataFrame, output_dir: str = "figs") -> str:
    """Grouped bar chart of output length by demographic group across models."""
    os.makedirs(output_dir, exist_ok=True)

    if df.empty:
        logger.warning("No data for demographic group plot")
        return ""

    # Normalise output length per model for fair comparison
    model_max = df.groupby("model")["avg_output_length"].transform("max")
    df = df.copy()
    df["normalised_length"] = df["avg_output_length"] / model_max * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x="model", y="normalised_length", hue="group", ax=ax)
    ax.set_title(
        "Output Length by Demographic Group (normalised per model)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Normalised Output Length (%)")
    ax.legend(
        title="Demographic Group",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, "demographic_groups.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 5: Parameter count vs bias ─────────────────────────────────────────


def plot_size_vs_bias(df: pd.DataFrame, output_dir: str = "figs") -> str:
    """Scatter plot of parameter count vs bias score, coloured by benchmark."""
    os.makedirs(output_dir, exist_ok=True)

    plot_df = df.dropna(subset=["score", "params"]).copy()
    if plot_df.empty:
        logger.warning("No data for size-vs-bias plot")
        return ""

    plot_df["params_m"] = plot_df["params"] / 1e6

    benchmarks = plot_df["benchmark"].unique()
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(7 * len(benchmarks), 5), squeeze=False)
    axes = axes[0]

    colors = sns.color_palette("Set2", len(benchmarks))

    for ax, benchmark, color in zip(axes, benchmarks, colors, strict=True):
        bdf = plot_df[plot_df["benchmark"] == benchmark]
        if bdf.empty:
            ax.set_title(f"{benchmark}\n(no data)")
            continue

        for _, row in bdf.iterrows():
            ax.scatter(
                row["params_m"],
                row["score"],
                s=100,
                color=color,
                zorder=5,
                edgecolors="white",
                linewidth=0.5,
            )
            ax.annotate(
                row["model"],
                (row["params_m"], row["score"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.8,
            )

        # Trend line
        if len(bdf) >= 3:
            z = np.polyfit(bdf["params_m"], bdf["score"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(bdf["params_m"].min(), bdf["params_m"].max(), 100)
            ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.5)

        ax.set_xlabel("Parameters (M)")
        ax.set_ylabel(BENCHMARK_LABELS.get(benchmark, benchmark))
        label = BENCHMARK_LABELS.get(benchmark, benchmark)
        ax.set_title(label.replace("\n", " ") if label else benchmark)

    plt.suptitle(
        "Model Size vs Bias Score",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    path = os.path.join(output_dir, "size_vs_bias.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── CLI entry point ─────────────────────────────────────────────────────────


def main() -> None:
    """Generate all cross-model visualisations."""
    parser = argparse.ArgumentParser(description="Cross-model benchmark visualisations")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="figs", help="Output directory")
    args = parser.parse_args()

    df = load_all_results(args.results_dir)
    if df.empty:
        print("No results found. Run benchmarks first.")
        return

    print(f"Found {len(df)} model x benchmark combinations")
    print(f"Models: {', '.join(sorted(df['model'].unique()))}")
    print(f"Benchmarks: {', '.join(sorted(df['benchmark'].unique()))}")
    print()

    # Plot 1: Heatmap
    path = plot_heatmap(df, args.output_dir)
    if path:
        print(f"1. Cross-model heatmap: {path}")

    # Plot 2: StereoSet categories
    cat_df = load_per_category_data(args.results_dir)
    path = plot_stereoset_categories(cat_df, args.output_dir)
    if path:
        print(f"2. StereoSet categories: {path}")

    # Plot 3: WinoBias pronouns
    pron_df = load_per_pronoun_data(args.results_dir)
    path = plot_winobias_pronouns(pron_df, args.output_dir)
    if path:
        print(f"3. WinoBias pronouns: {path}")

    # Plot 4: Demographic groups
    demo_df = load_per_group_data(args.results_dir)
    path = plot_demographic_groups(demo_df, args.output_dir)
    if path:
        print(f"4. Demographic groups: {path}")

    # Plot 5: Size vs bias
    path = plot_size_vs_bias(df, args.output_dir)
    if path:
        print(f"5. Size vs bias: {path}")

    print(f"\nDone. Figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
