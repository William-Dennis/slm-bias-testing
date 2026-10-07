"""Tests for slm_bias_testing.analysis — statistical helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from slm_bias_testing.analysis import (
    build_summary_table,
    cohens_d,
    collapse_runs,
    group_summary,
    pairwise_comparisons,
    per_cv_variance,
    variance_breakdown,
)


def repeated_runs(groups: dict[str, list[float]], runs: int = 3) -> pd.DataFrame:
    """Build a records-like frame: one CV per value in ``groups[group]``."""
    rows = []
    for group, means in groups.items():
        for i, mean in enumerate(means):
            for run in range(runs):
                rows.append(
                    {
                        "key": f"{group}-{i}",
                        "run": run,
                        "score": mean + (run - 1) * 0.5,
                        "group": group,
                    }
                )
    return pd.DataFrame(rows)


class TestCohensD:
    def test_identical_groups(self):
        s1 = pd.Series([10, 10, 10])
        s2 = pd.Series([10, 10, 10])
        assert cohens_d(s1, s2) == 0.0

    def test_different_groups(self):
        s1 = pd.Series([10, 12, 14])
        s2 = pd.Series([20, 22, 24])
        d = cohens_d(s1, s2)
        assert d < 0  # s1 < s2

    def test_small_sample(self):
        s1 = pd.Series([10])
        s2 = pd.Series([20])
        assert cohens_d(s1, s2) == 0.0  # n < 2

    def test_zero_pooled_std(self):
        s1 = pd.Series([5, 5])
        s2 = pd.Series([5, 5])
        assert cohens_d(s1, s2) == 0.0


class TestGroupSummary:
    def test_basic(self, sample_df):
        result = group_summary(sample_df, "name")
        assert not result.empty
        assert "mean" in result.columns
        assert "count" in result.columns

    def test_missing_column(self, sample_df):
        result = group_summary(sample_df, "nonexistent")
        assert result.empty

    def test_all_nan_column(self):
        df = pd.DataFrame({"group": [None, None], "score": [1, 2]})
        result = group_summary(df, "group")
        assert result.empty


class TestPairwiseComparisons:
    def test_basic(self):
        df = repeated_runs({"Alice": [80, 81, 82], "Bob": [60, 61, 62]})
        result = pairwise_comparisons(df, "group")
        assert not result.empty
        assert "cohens_d" in result.columns
        assert "p_value" in result.columns
        assert "p_holm" in result.columns

    def test_single_group(self):
        df = pd.DataFrame({"group": ["A", "A"], "score": [1, 2]})
        result = pairwise_comparisons(df, "group")
        assert result.empty

    def test_missing_column(self, sample_df):
        result = pairwise_comparisons(sample_df, "nonexistent")
        assert result.empty

    def test_sign_independent_of_row_order(self):
        df = repeated_runs({"A": [90, 91, 92], "B": [10, 11, 12]})
        forward = pairwise_comparisons(df, "group")
        shuffled = pairwise_comparisons(
            df.sample(frac=1, random_state=7).reset_index(drop=True), "group"
        )
        row_f = forward.iloc[0]
        row_s = shuffled.iloc[0]
        assert row_f["group1"] == row_s["group1"] == "A"  # sorted, not appearance order
        assert row_f["cohens_d"] == row_s["cohens_d"]
        assert row_f["p_value"] == row_s["p_value"]

    def test_holm_correction_properties(self):
        # A/B nearly identical (large raw p), C far away (tiny raw p).
        df = repeated_runs(
            {
                "A": [50.0, 50.1, 49.9, 50.05],
                "B": [50.2, 50.0, 50.1, 50.0],
                "C": [10.0, 10.1, 9.9, 10.05],
            }
        )
        result = pairwise_comparisons(df, "group").set_index(["group1", "group2"])
        assert (result["p_holm"] >= result["p_value"] - 1e-12).all()
        assert (result["p_holm"] <= 1.0).all()
        # Largest raw p keeps its value (Holm multiplier 1, cummax from below).
        ab = result.loc[("A", "B")]
        assert ab["p_holm"] == pytest.approx(ab["p_value"], abs=1e-6)
        # Smallest raw p is multiplied by the number of pairs (3).
        smallest = result["p_value"].idxmin()
        assert result.loc[smallest, "p_holm"] == pytest.approx(
            min(1.0, 3 * result.loc[smallest, "p_value"]), abs=1e-9
        )

    def test_counts_cvs_not_runs(self):
        df = repeated_runs({"A": [70, 71], "B": [30, 31]}, runs=5)
        result = pairwise_comparisons(df, "group")
        assert result["n1"].tolist() == [2]  # 2 CVs, not 10 rows
        assert result["n2"].tolist() == [2]

    def test_row_level_without_key_column(self):
        df = pd.DataFrame(
            {"group": ["A"] * 4 + ["B"] * 4, "score": [70, 71, 72, 73, 30, 31, 32, 33]}
        )
        result = pairwise_comparisons(df, "group")
        assert result["n1"].tolist() == [4]  # no key → rows are the units

    def test_undefined_test_does_not_poison_holm(self):
        # A and B are identical constant groups → Welch p is NaN; that
        # undefined pair must not force every other pair's p_holm to 1.0.
        df = repeated_runs({"A": [70.0, 70.0], "B": [70.0, 70.0], "C": [10.0, 11.0]})
        result = pairwise_comparisons(df, "group").set_index(["group1", "group2"])
        ab = result.loc[("A", "B")]
        assert ab["p_value"] != ab["p_value"]  # NaN
        assert ab["p_holm"] != ab["p_holm"]  # undefined stays undefined
        for pair in [("A", "C"), ("B", "C")]:
            assert result.loc[pair, "p_holm"] < 1.0
            assert result.loc[pair, "p_holm"] >= result.loc[pair, "p_value"]


class TestVarianceBreakdown:
    def test_basic(self, sample_df):
        result = variance_breakdown(sample_df, ["name", "university"])
        assert "name" in result
        assert "proportion" in result["name"]

    def test_zero_total_variance(self):
        df = pd.DataFrame({"factor": ["A", "B"], "score": [5.0, 5.0]})
        result = variance_breakdown(df, ["factor"])
        assert result == {}

    def test_missing_column(self, sample_df):
        result = variance_breakdown(sample_df, ["nonexistent"])
        assert "nonexistent" not in result

    def test_tiny_proportion_not_rounded_to_zero(self):
        # Between-group variance is ~1e-6 of total; display rounding would
        # zero it, but machine-readable output must keep the raw value.
        df = pd.DataFrame({"factor": ["A", "A", "B", "B"], "score": [0.0, 1000.0, 1.0, 1001.0]})
        proportion = variance_breakdown(df, ["factor"])["factor"]["proportion"]
        assert 0 < proportion < 0.001


class TestCollapseRuns:
    def test_collapses_to_one_row_per_cv(self):
        df = repeated_runs({"A": [70, 80]}, runs=4)
        collapsed = collapse_runs(df, ["group"])
        assert len(collapsed) == 2
        # runs=4 adds (run-1)*0.5 jitter {-0.5, 0, 0.5, 1.0} → mean +0.25
        assert sorted(collapsed["score"]) == [70.25, 80.25]
        assert set(collapsed["group"]) == {"A"}

    def test_identity_without_key_column(self):
        df = pd.DataFrame({"group": ["A"], "score": [1.0]})
        assert collapse_runs(df, ["group"]) is df

    def test_identity_without_score_column(self):
        df = pd.DataFrame({"key": ["a"], "group": ["A"]})
        assert collapse_runs(df, ["group"]) is df

    def test_identity_when_no_group_columns_present(self):
        df = pd.DataFrame({"key": ["a", "a"], "score": [1.0, 2.0]})
        assert collapse_runs(df, ["missing"]) is df

    def test_null_keys_kept_as_singleton_cvs(self, caplog):
        # groupby drops null keys by default — those rows must not vanish.
        df = pd.DataFrame(
            {
                "key": ["k1", "k1", None],
                "run": [0, 1, 0],
                "score": [70.0, 70.0, 90.0],
                "group": ["A", "A", "A"],
            }
        )
        result = collapse_runs(df, ["group"])
        assert len(result) == 2  # k1 collapsed + singleton null-key row
        assert sorted(result["score"]) == [70.0, 90.0]
        assert "singleton CV" in caplog.text

    def test_null_keys_do_not_shrink_group_counts(self):
        df = pd.DataFrame(
            {
                "key": ["k1", "k1", None],
                "run": [0, 1, 0],
                "score": [70.0, 70.0, 90.0],
                "group": ["A", "A", "A"],
            }
        )
        assert group_summary(df, "group").loc["A", "count"] == 2


class TestClusterAwareInference:
    def test_group_summary_ci_uses_cv_means(self):
        # Two CVs in group A: constant 80 and constant 70 across runs.
        df = pd.DataFrame(
            {
                "key": ["k1", "k1", "k1", "k2", "k2", "k2"],
                "run": [0, 1, 2, 0, 1, 2],
                "score": [80.0, 80.0, 80.0, 70.0, 70.0, 70.0],
                "group": ["A"] * 6,
            }
        )
        result = group_summary(df, "group").loc["A"]
        # CV means are 80 and 70: n = 2 CVs (NOT 6 rows), mean 75.
        assert result["count"] == 2
        assert result["mean"] == 75.0
        # se = sqrt(50)/sqrt(2) = 5; t(1, .975) = 12.7062; CI = 75 +/- 63.531
        assert result["ci_lower"] == pytest.approx(11.469, abs=0.01)
        assert result["ci_upper"] == pytest.approx(138.531, abs=0.01)

    def test_group_summary_counts_cvs(self, sample_df):
        # sample_df: two CVs (a=Alice, b=Bob), three runs each.
        result = group_summary(sample_df, "name")
        assert result["count"].tolist() == [1, 1]

    def test_group_summary_row_level_without_key(self):
        df = pd.DataFrame({"group": ["A", "A", "A"], "score": [10.0, 20.0, 30.0]})
        result = group_summary(df, "group").loc["A"]
        assert result["count"] == 3

    def test_variance_breakdown_over_cv_means(self):
        # Within-CV run noise is huge but identical across CVs; collapsing
        # removes it from the denominator, so the factor explains ~50%.
        rows = []
        for key, base, group in [
            ("k1", 10.0, "A"),
            ("k2", 12.0, "A"),
            ("k3", 90.0, "B"),
            ("k4", 92.0, "B"),
        ]:
            for run, jitter in enumerate([-25.0, 25.0]):
                rows.append({"key": key, "run": run, "score": base + jitter, "group": group})
        df = pd.DataFrame(rows)
        proportion = variance_breakdown(df, ["group"])["group"]["proportion"]
        assert proportion > 0.9  # run noise collapsed away; group dominates


class TestPerCvVariance:
    def test_basic(self, sample_df):
        cv_std, summary = per_cv_variance(sample_df, "key", "score")
        assert len(cv_std) > 0
        assert "mean_cv_std" in summary

    def test_missing_key_column(self):
        df = pd.DataFrame({"score": [1, 2, 3]})
        cv_std, summary = per_cv_variance(df)
        assert cv_std.empty
        assert summary == {}


class TestBuildSummaryTable:
    def test_returns_string(self, sample_df):
        result = build_summary_table(sample_df, ["name", "university"])
        assert isinstance(result, str)
        assert "STATISTICAL ANALYSIS" in result

    def test_empty_dataframe(self):
        df = pd.DataFrame({"score": pd.Series(dtype=float)})
        result = build_summary_table(df, [])
        assert "STATISTICAL ANALYSIS" in result
