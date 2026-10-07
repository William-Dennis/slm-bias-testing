"""Statistical analysis for CV screening benchmark.

Repeated runs of the same CV are *not* independent observations. Every
inference helper here (CIs, t-tests, effect sizes, variance breakdown)
therefore collapses repeated runs to one observation per CV (mean across
runs) before computing statistics — the CV is the sampling unit. Frames
without a ``key`` column are treated as already independent.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def collapse_runs(
    df: pd.DataFrame,
    group_cols: list[str],
    score_col: str = "score",
    key_col: str = "key",
) -> pd.DataFrame:
    """Collapse repeated runs to one row per CV (mean score across runs).

    Returns a frame with one row per ``key``: the mean score plus the first
    value of each requested group column. Falls back to the input frame
    unchanged when there is no key column (rows are already independent) or
    when grouping by the key itself. Rows with a null key cannot be grouped
    by pandas (``groupby`` drops them), so each is treated as its own
    singleton CV and a warning is logged — no scored row is ever dropped.
    """
    if df.empty or score_col not in df.columns or key_col not in df.columns:
        return df
    keep = [c for c in group_cols if c in df.columns and c != key_col]
    if not keep:
        return df
    if df[key_col].isna().any():
        logger.warning(
            "%d row(s) have a null %s; treating each as its own singleton CV",
            int(df[key_col].isna().sum()),
            key_col,
        )
        df = df.copy()
        null_mask = df[key_col].isna()
        df.loc[null_mask, key_col] = [f"__null_{i}" for i in df.index[null_mask]]
    agg: dict[str, str] = {score_col: "mean"}
    agg.update({col: "first" for col in keep})
    return df.groupby(key_col, sort=True).agg(agg).reset_index(drop=True)


def _for_inference(
    df: pd.DataFrame, group_cols: list[str], score_col: str, key_col: str | None
) -> pd.DataFrame:
    """Frame ready for inference: collapsed to CV-level when clustering applies."""
    if key_col is None:
        return df
    return collapse_runs(df, group_cols, score_col, key_col)


def group_summary(
    df: pd.DataFrame,
    group_col: str,
    score_col: str = "score",
    cluster_col: str | None = "key",
) -> pd.DataFrame:
    """Mean, std, count, and 95% CI per group — over CVs, not raw runs.

    Repeated runs are collapsed with :func:`collapse_runs` first, so
    ``count`` is the number of independent CVs and the CI reflects between-CV
    spread instead of treating every run as an independent sample.
    """
    if group_col not in df.columns or df[group_col].isna().all():
        return pd.DataFrame()
    df = _for_inference(df, [group_col], score_col, cluster_col)
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
    df: pd.DataFrame,
    group_col: str,
    score_col: str = "score",
    cluster_col: str | None = "key",
) -> pd.DataFrame:
    """Cohen's d and Welch t-test for all pairs of groups, with Holm correction.

    Groups are ordered by string value so results do not depend on row
    order; ``cohens_d`` is ``mean(group1) - mean(group2)`` for
    ``group1 < group2``, making the sign stable across runs. Repeated runs
    are collapsed to per-CV means first (see :func:`collapse_runs`), and
    ``p_holm`` holds Holm-Bonferroni adjusted p-values across every pair
    with a defined test (undefined Welch tests — NaN p — are excluded from
    the correction and reported as null ``p_holm``); ``p_value`` stays the
    raw Welch p.
    """
    if group_col not in df.columns:
        return pd.DataFrame()
    df = _for_inference(df, [group_col], score_col, cluster_col)
    groups = sorted(df[group_col].dropna().unique(), key=str)
    if len(groups) < 2:
        return pd.DataFrame()
    # Pre-split to avoid redundant boolean masks per pair
    grouped = {g: df.loc[df[group_col] == g, score_col].dropna() for g in groups}
    rows: list[dict[str, Any]] = []
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
                    "p_value": float(p_val),
                    "mean1": round(g1.mean(), 2),
                    "mean2": round(g2.mean(), 2),
                    "n1": len(g1),
                    "n2": len(g2),
                }
            )

    # Holm-Bonferroni over the finite (defined) tests only: an undefined
    # Welch test (NaN p, e.g. two identical constant groups) must not poison
    # the running maximum for every other pair.
    finite = [
        (idx, rows[idx]["p_value"])
        for idx in range(len(rows))
        if not np.isnan(rows[idx]["p_value"])
    ]
    n_finite = len(finite)
    running = 0.0
    adjusted: dict[int, float] = {}
    for rank, (idx, p_value) in enumerate(sorted(finite, key=lambda item: item[1])):
        running = max(running, min(1.0, (n_finite - rank) * p_value))
        adjusted[idx] = running
    for idx, row in enumerate(rows):
        row["p_value"] = round(row["p_value"], 4)
        row["p_holm"] = round(adjusted[idx], 4) if idx in adjusted else float("nan")
    return pd.DataFrame(rows)


def variance_breakdown(
    df: pd.DataFrame,
    factors: list[str],
    score_col: str = "score",
    cluster_col: str | None = "key",
) -> dict[str, Any]:
    """Proportion of total variance explained by each factor.

    Computed over per-CV means (repeated runs collapsed — see
    :func:`collapse_runs`) so within-CV sampling noise does not inflate the
    denominator. Values are raw floats; round for display only.
    """
    if score_col not in df.columns:
        return {}
    df = _for_inference(df, factors, score_col, cluster_col)
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
            "variance_explained": float(between_var),
            "proportion": float(between_var / total_var),
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
    n_cvs = df["key"].nunique() if "key" in df.columns else len(df)
    lines.append(f"  N rows (CV x run): {len(df)}")
    lines.append(f"  N CVs (independent units): {n_cvs}")
    lines.append(f"  Overall mean score: {df[score_col].mean():.2f}")
    lines.append(f"  Overall std: {df[score_col].std():.2f}")
    lines.append("  Note: CIs, t-tests and effect sizes below are computed over")
    lines.append("  per-CV means (repeated runs collapsed), not raw runs.")

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
            lines.append("\nPairwise comparisons (p_holm = Holm-Bonferroni adjusted):")
            display_cols = [
                "group1",
                "group2",
                "cohens_d",
                "p_value",
                "p_holm",
                "mean1",
                "mean2",
                "n1",
                "n2",
            ]
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
