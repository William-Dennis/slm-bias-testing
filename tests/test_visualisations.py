"""Tests for slm_bias_testing.visualisations — data loading and plotting."""

from __future__ import annotations

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

matplotlib.use("Agg")

from slm_bias_testing.visualisations import (
    load_all_results,
    load_per_category_data,
    load_per_group_data,
    load_per_pronoun_data,
    plot_demographic_groups,
    plot_heatmap,
    plot_size_vs_bias,
    plot_stereoset_categories,
    plot_winobias_pronouns,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _close_all_figures():
    yield
    plt.close("all")


# ── Helper factories ─────────────────────────────────────────────────


def _stereoset_json(score=5.0, per_category=None):
    return {
        "overall_stereotype_score": score,
        "n_examples": 100,
        "per_category": per_category
        or {
            "gender": 4.0,
            "race": 6.0,
            "religion": 5.0,
            "profession": 5.0,
        },
    }


def _winobias_json(bias_score=7.2, per_pronoun=None):
    return {
        "bias_score": bias_score,
        "n_examples": 100,
        "per_pronoun": per_pronoun
        or {
            "he": 72.0,
            "she": 68.0,
            "they": 70.0,
        },
    }


def _demographic_bias_json(male_len=150.0, female_len=120.0, per_group=None):
    groups = per_group or {
        "gender_male": {"n": 50, "avg_output_length": male_len},
        "gender_female": {"n": 50, "avg_output_length": female_len},
        "race_asian": {"n": 25, "avg_output_length": 130.0},
        "race_black": {"n": 25, "avg_output_length": 125.0},
    }
    return {
        "per_group": groups,
        "n_examples": 200,
    }


def _cv_screening_json(mean_score=82.5):
    return {
        "mean_score": mean_score,
        "n_examples": 600,
    }


def _setup_model_dir(base_dir, model, benchmarks=None):
    if benchmarks is None:
        benchmarks = ["stereoset", "winobias", "demographic-bias", "cv-screening"]

    data_map = {
        "stereoset": _stereoset_json,
        "winobias": _winobias_json,
        "demographic-bias": _demographic_bias_json,
        "cv-screening": _cv_screening_json,
    }

    for bm in benchmarks:
        bm_dir = os.path.join(base_dir, model, bm)
        os.makedirs(bm_dir, exist_ok=True)
        data = data_map[bm]()
        with open(os.path.join(bm_dir, f"{bm}.json"), "w") as f:
            json.dump(data, f)


@pytest.fixture()
def mock_results_dir(tmp_path):
    _setup_model_dir(str(tmp_path), "smollm-135m")
    _setup_model_dir(str(tmp_path), "smollm2-135m")
    return str(tmp_path)


# ── Data loading tests ───────────────────────────────────────────────


class TestLoadAllResults:
    def test_empty_directory(self, tmp_path):
        df = load_all_results(str(tmp_path))
        assert df.empty

    def test_single_model_all_benchmarks(self, mock_results_dir):
        df = load_all_results(mock_results_dir)
        assert len(df) == 8
        assert set(df["model"].unique()) == {"smollm-135m", "smollm2-135m"}
        assert set(df["benchmark"].unique()) == {
            "stereoset",
            "winobias",
            "demographic-bias",
            "cv-screening",
        }

    def test_scores_are_float(self, mock_results_dir):
        df = load_all_results(mock_results_dir)
        assert df["score"].dtype == float

    def test_includes_registry_metadata(self, mock_results_dir):
        df = load_all_results(mock_results_dir)
        row = df[(df["model"] == "smollm-135m") & (df["benchmark"] == "stereoset")].iloc[0]
        assert row["params"] == 135_000_000
        assert row["family"] == "huggingface"
        assert row["release_date"] == "2024-07"

    def test_unknown_model_gets_empty_metadata(self, tmp_path):
        _setup_model_dir(str(tmp_path), "unknown-model", benchmarks=["stereoset"])
        df = load_all_results(str(tmp_path))
        assert df.iloc[0]["family"] == "unknown"

    def test_missing_result_file_skipped(self, tmp_path):
        os.makedirs(os.path.join(str(tmp_path), "model-a", "stereoset"))
        df = load_all_results(str(tmp_path))
        assert df.empty

    def test_malformed_json_skipped(self, tmp_path):
        bm_dir = os.path.join(str(tmp_path), "model-a", "stereoset")
        os.makedirs(bm_dir)
        with open(os.path.join(bm_dir, "stereoset.json"), "w") as f:
            f.write("{bad json")
        df = load_all_results(str(tmp_path))
        assert df.empty

    def test_hidden_directories_ignored(self, tmp_path):
        _setup_model_dir(str(tmp_path), ".hidden", benchmarks=["stereoset"])
        df = load_all_results(str(tmp_path))
        assert df.empty

    def test_demographic_bias_score_is_computed(self, mock_results_dir):
        df = load_all_results(mock_results_dir)
        demo = df[df["benchmark"] == "demographic-bias"]
        assert len(demo) == 2
        # abs(150-120)/max(150,120)*100 = 30/150*100 = 20.0
        assert demo.iloc[0]["score"] == pytest.approx(20.0, abs=0.01)

    def test_demographic_bias_zero_length_returns_none(self, tmp_path):
        bm_dir = os.path.join(str(tmp_path), "smollm-135m", "demographic-bias")
        os.makedirs(bm_dir)
        with open(os.path.join(bm_dir, "demographic-bias.json"), "w") as f:
            json.dump(
                {
                    "per_group": {
                        "gender_male": {"n": 50, "avg_output_length": 150},
                        "gender_female": {"n": 50, "avg_output_length": 0},
                    }
                },
                f,
            )
        df = load_all_results(str(tmp_path))
        assert df.iloc[0]["score"] is None

    def test_missing_score_field_returns_none(self, tmp_path):
        bm_dir = os.path.join(str(tmp_path), "model-a", "stereoset")
        os.makedirs(bm_dir)
        with open(os.path.join(bm_dir, "stereoset.json"), "w") as f:
            json.dump({"n_examples": 10}, f)
        df = load_all_results(str(tmp_path))
        assert df.iloc[0]["score"] is None


class TestLoadPerGroupData:
    def test_empty_directory(self, tmp_path):
        df = load_per_group_data(str(tmp_path))
        assert df.empty

    def test_returns_per_group_rows(self, mock_results_dir):
        df = load_per_group_data(mock_results_dir)
        assert len(df) == 8  # 2 models x 4 groups
        assert "gender_male" in df["group"].values
        assert "race_black" in df["group"].values
        assert df["avg_output_length"].dtype == float

    def test_missing_file_skipped(self, tmp_path):
        os.makedirs(os.path.join(str(tmp_path), "model-a", "demographic-bias"))
        df = load_per_group_data(str(tmp_path))
        assert df.empty

    def test_malformed_json_skipped(self, tmp_path):
        bm_dir = os.path.join(str(tmp_path), "model-a", "demographic-bias")
        os.makedirs(bm_dir)
        with open(os.path.join(bm_dir, "demographic-bias.json"), "w") as f:
            f.write("{bad")
        df = load_per_group_data(str(tmp_path))
        assert df.empty

    def test_empty_per_group(self, tmp_path):
        bm_dir = os.path.join(str(tmp_path), "model-a", "demographic-bias")
        os.makedirs(bm_dir)
        with open(os.path.join(bm_dir, "demographic-bias.json"), "w") as f:
            json.dump({"per_group": {}}, f)
        df = load_per_group_data(str(tmp_path))
        assert df.empty


class TestLoadPerCategoryData:
    def test_empty_directory(self, tmp_path):
        df = load_per_category_data(str(tmp_path))
        assert df.empty

    def test_returns_per_category_rows(self, mock_results_dir):
        df = load_per_category_data(mock_results_dir)
        assert len(df) == 8  # 2 models x 4 categories
        assert set(df["category"].unique()) == {"gender", "race", "religion", "profession"}
        assert df["score"].dtype == float

    def test_missing_file_skipped(self, tmp_path):
        os.makedirs(os.path.join(str(tmp_path), "model-a", "stereoset"))
        df = load_per_category_data(str(tmp_path))
        assert df.empty

    def test_empty_per_category(self, tmp_path):
        bm_dir = os.path.join(str(tmp_path), "model-a", "stereoset")
        os.makedirs(bm_dir)
        with open(os.path.join(bm_dir, "stereoset.json"), "w") as f:
            json.dump({"per_category": {}}, f)
        df = load_per_category_data(str(tmp_path))
        assert df.empty


class TestLoadPerPronounData:
    def test_empty_directory(self, tmp_path):
        df = load_per_pronoun_data(str(tmp_path))
        assert df.empty

    def test_returns_per_pronoun_rows(self, mock_results_dir):
        df = load_per_pronoun_data(mock_results_dir)
        assert len(df) == 6  # 2 models x 3 pronouns
        assert set(df["pronoun"].unique()) == {"he", "she", "they"}
        assert df["accuracy"].dtype == float

    def test_missing_file_skipped(self, tmp_path):
        os.makedirs(os.path.join(str(tmp_path), "model-a", "winobias"))
        df = load_per_pronoun_data(str(tmp_path))
        assert df.empty

    def test_empty_per_pronoun(self, tmp_path):
        bm_dir = os.path.join(str(tmp_path), "model-a", "winobias")
        os.makedirs(bm_dir)
        with open(os.path.join(bm_dir, "winobias.json"), "w") as f:
            json.dump({"per_pronoun": {}}, f)
        df = load_per_pronoun_data(str(tmp_path))
        assert df.empty


# ── Plotting tests ───────────────────────────────────────────────────


class TestPlotHeatmap:
    def test_empty_dataframe_returns_empty_string(self, tmp_path):
        df = pd.DataFrame(columns=pd.Index(["model", "benchmark", "score"]))
        path = plot_heatmap(df, str(tmp_path))
        assert path == ""

    def test_creates_png_file(self, mock_results_dir, tmp_path):
        df = load_all_results(mock_results_dir)
        path = plot_heatmap(df, str(tmp_path))
        assert path.endswith("cross_model_heatmap.png")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_creates_output_directory(self, mock_results_dir, tmp_path):
        output = os.path.join(str(tmp_path), "nested", "subdir")
        df = load_all_results(mock_results_dir)
        path = plot_heatmap(df, output)
        assert os.path.isfile(path)


class TestPlotStereoSetCategories:
    def test_empty_dataframe_returns_empty_string(self, tmp_path):
        df = pd.DataFrame(columns=pd.Index(["model", "category", "score"]))
        path = plot_stereoset_categories(df, str(tmp_path))
        assert path == ""

    def test_creates_png_file(self, mock_results_dir, tmp_path):
        df = load_per_category_data(mock_results_dir)
        path = plot_stereoset_categories(df, str(tmp_path))
        assert path.endswith("stereoset_categories.png")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


class TestPlotWinoBiasPronouns:
    def test_empty_dataframe_returns_empty_string(self, tmp_path):
        df = pd.DataFrame(columns=pd.Index(["model", "pronoun", "accuracy"]))
        path = plot_winobias_pronouns(df, str(tmp_path))
        assert path == ""

    def test_creates_png_file(self, mock_results_dir, tmp_path):
        df = load_per_pronoun_data(mock_results_dir)
        path = plot_winobias_pronouns(df, str(tmp_path))
        assert path.endswith("winobias_pronouns.png")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


class TestPlotDemographicGroups:
    def test_empty_dataframe_returns_empty_string(self, tmp_path):
        df = pd.DataFrame(columns=pd.Index(["model", "group", "n", "avg_output_length"]))
        path = plot_demographic_groups(df, str(tmp_path))
        assert path == ""

    def test_creates_png_file(self, mock_results_dir, tmp_path):
        df = load_per_group_data(mock_results_dir)
        path = plot_demographic_groups(df, str(tmp_path))
        assert path.endswith("demographic_groups.png")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


class TestPlotSizeVsBias:
    def test_empty_dataframe_returns_empty_string(self, tmp_path):
        df = pd.DataFrame(columns=pd.Index(["model", "benchmark", "score", "params"]))
        path = plot_size_vs_bias(df, str(tmp_path))
        assert path == ""

    def test_all_nan_values_returns_empty_string(self, tmp_path):
        df = pd.DataFrame(
            {
                "model": ["a", "b"],
                "benchmark": ["stereoset", "stereoset"],
                "score": [None, None],
                "params": [100_000_000, 200_000_000],
            }
        )
        path = plot_size_vs_bias(df, str(tmp_path))
        assert path == ""

    def test_creates_png_file(self, mock_results_dir, tmp_path):
        df = load_all_results(mock_results_dir)
        path = plot_size_vs_bias(df, str(tmp_path))
        assert path.endswith("size_vs_bias.png")
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
