"""Statistical analysis for CV screening benchmark."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def group_summary(df: pd.DataFrame, group_col: str, score_col: str = "score") -> pd.DataFrame:
    """Mean, std, count, and 95% CI per group."""
    if group_col not in df.columns or df[group_col].isna().all():
        return pd.DataFrame()
    groups = df.groupby(group_col)[score_col]
    summary = groups.agg(["mean", "std", "count"])
    confidence = 0.95
    ci_lower = []
    ci_upper = []
    for idx in summary.index:
        n = summary.loc[idx, "count"]
        mean = summary.loc[idx, "mean"]
        std = summary.loc[idx, "std"]
        se = std / np.sqrt(n)
        t_val = sp_stats.t.ppf((1 + confidence) / 2, n - 1) if n > 1 else 0
        ci_lower.append(mean - t_val * se)
        ci_upper.append(mean + t_val * se)
    summary["ci_lower"] = ci_lower
    summary["ci_upper"] = ci_upper
    return summary


def cohens_d(series1: pd.Series, series2: pd.Series) -> float:
    """Cohen's d for two independent groups (pooled standard deviation)."""
    n1, n2 = len(series1), len(series2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = series1.std(ddof=1), series2.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (series1.mean() - series2.mean()) / pooled  # type: ignore[no-any-return]


def pairwise_comparisons(
    df: pd.DataFrame, group_col: str, score_col: str = "score"
) -> pd.DataFrame:
    """Cohen's d and t-test for all pairs of groups."""
    if group_col not in df.columns:
        return pd.DataFrame()
    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        return pd.DataFrame()
    # Pre-split to avoid redundant boolean masks per pair
    grouped = {g: df.loc[df[group_col] == g, score_col].dropna() for g in groups}
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = grouped[groups[i]]
            g2 = grouped[groups[j]]
            if len(g1) < 2 or len(g2) < 2:
                continue
            d = cohens_d(g1, g2)
            t_stat, p_val = sp_stats.ttest_ind(g1, g2, equal_var=False)
            rows.append(
                {
                    "group_col": group_col,
                    "group1": groups[i],
                    "group2": groups[j],
                    "cohens_d": round(d, 3),
                    "t_statistic": round(t_stat, 3),
                    "p_value": round(p_val, 4),
                    "mean1": round(g1.mean(), 2),
                    "mean2": round(g2.mean(), 2),
                    "n1": len(g1),
                    "n2": len(g2),
                }
            )
    return pd.DataFrame(rows)


def variance_breakdown(
    df: pd.DataFrame, factors: list[str], score_col: str = "score"
) -> dict[str, Any]:
    """Proportion of total variance explained by each factor."""
    if score_col not in df.columns:
        return {}
    total_var = df[score_col].var(ddof=0)
    if total_var == 0:
        return {}
    results = {}
    grand_mean = df[score_col].mean()
    n_total = len(df)
    for factor in factors:
        if factor not in df.columns:
            continue
        group_means = df.groupby(factor)[score_col].mean()
        group_counts = df.groupby(factor)[score_col].count()
        between_var = (group_counts * (group_means - grand_mean) ** 2).sum() / n_total
        results[factor] = {
            "variance_explained": round(between_var, 3),
            "proportion": round(between_var / total_var, 4),
        }
    return results


def per_cv_variance(
    df: pd.DataFrame, key_col: str = "key", score_col: str = "score"
) -> tuple[pd.Series, dict[str, Any]]:
    """Std deviation per CV across runs, plus overall summary."""
    if key_col not in df.columns:
        return pd.Series(dtype=float), {}
    cv_std = df.groupby(key_col)[score_col].std().dropna()
    summary = {}
    if len(cv_std) > 0:
        summary = {
            "mean_cv_std": cv_std.mean(),
            "median_cv_std": cv_std.median(),
            "min_cv_std": cv_std.min(),
            "max_cv_std": cv_std.max(),
            "p25_cv_std": cv_std.quantile(0.25),
            "p75_cv_std": cv_std.quantile(0.75),
        }
    return cv_std, summary


def build_summary_table(df: pd.DataFrame, group_cols: list[str], score_col: str = "score") -> str:
    """Build formatted summary string with group means, CI, effect sizes."""
    lines = []
    lines.append("=" * 90)
    lines.append("STATISTICAL ANALYSIS")
    lines.append("=" * 90)

    _cv_std, cv_summary = per_cv_variance(df, key_col="key", score_col=score_col)
    if cv_summary:
        lines.append("\n--- Per-CV Variance (std across runs) ---")
        lines.append(f"  Mean within-CV std: {cv_summary['mean_cv_std']:.3f}")
        lines.append(f"  Median within-CV std: {cv_summary['median_cv_std']:.3f}")
        lines.append(f"  Min within-CV std: {cv_summary['min_cv_std']:.3f}")
        lines.append(f"  Max within-CV std: {cv_summary['max_cv_std']:.3f}")
        lines.append(
            f"  25th-75th percentile: {cv_summary['p25_cv_std']:.3f} - {cv_summary['p75_cv_std']:.3f}"
        )

    lines.append("\n--- Overall ---")
    lines.append(f"  N observations: {len(df)}")
    lines.append(f"  Overall mean score: {df[score_col].mean():.2f}")
    lines.append(f"  Overall std: {df[score_col].std():.2f}")

    for col in group_cols:
        if col not in df.columns or df[col].isna().all():
            continue
        lines.append(f"\n{'─' * 90}")
        lines.append(f"Group: {col}")
        lines.append(f"{'─' * 90}")

        summary = group_summary(df, col, score_col)
        if not summary.empty:
            lines.append(summary.to_string())

        pw = pairwise_comparisons(df, col, score_col)
        if not pw.empty:
            lines.append("\nPairwise comparisons:")
            display_cols = ["group1", "group2", "cohens_d", "p_value", "mean1", "mean2", "n1", "n2"]
            lines.append(pw[display_cols].to_string(index=False))

    # Variance breakdown
    valid_factors = [c for c in group_cols if c in df.columns and not df[c].isna().all()]
    if valid_factors:
        lines.append(f"\n{'─' * 90}")
        lines.append("Variance Explained by Each Factor")
        lines.append(f"{'─' * 90}")
        vb = variance_breakdown(df, valid_factors, score_col)
        for factor, vals in vb.items():
            lines.append(
                f"  {factor}: variance = {vals['variance_explained']:.3f}, "
                f"proportion = {vals['proportion']:.3f}"
            )

    return "\n".join(lines)
