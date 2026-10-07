import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd

from slm_bias_testing.benchmark_runner import (
    _build_benchmark_summary,
    _write_summary,
    main,
    run_model_benchmarks,
)
from slm_bias_testing.registry import MODELS


class TestRunModelBenchmarks:
    @patch("slm_bias_testing.model_clients.OllamaPoolClient")
    @patch("slm_bias_testing.benchmark_runner.pull_model", return_value=True)
    @patch("slm_bias_testing.benchmark_runner.get_model")
    def test_skip_existing_results(self, mock_get_model, mock_pull, mock_pool):
        mock_get_model.return_value = {"ollama_tag": "smollm:135m"}

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "smollm-135m", "cv-screening")
            os.makedirs(results_dir)
            results_file = os.path.join(results_dir, "results.json")
            with open(results_file, "w") as f:
                json.dump({"model": "smollm-135m"}, f)

            run_model_benchmarks(
                "smollm-135m",
                "cv-screening",
                tmpdir,
            )

            # pull_model is called once (before loop), but benchmark is skipped
            mock_pull.assert_called_once()

    @patch("slm_bias_testing.model_clients.OllamaPoolClient")
    @patch("slm_bias_testing.benchmark_runner.pull_model", return_value=False)
    @patch("slm_bias_testing.benchmark_runner.get_model")
    def test_skip_on_pull_failure(self, mock_get_model, mock_pull, mock_pool):
        mock_get_model.return_value = {"ollama_tag": "smollm:135m"}

        with tempfile.TemporaryDirectory() as tmpdir:
            run_model_benchmarks(
                "smollm-135m",
                "cv-screening",
                tmpdir,
            )

            # Results dir should not be created
            results_dir = os.path.join(tmpdir, "smollm-135m", "cv-screening")
            assert not os.path.exists(results_dir)


class TestDispatch:
    """Benchmark dispatch, summary writing, and pool fallback (all mocked)."""

    @patch("slm_bias_testing.cv_screening.run_cv_screening")
    @patch("slm_bias_testing.model_clients.OllamaPoolClient")
    @patch("slm_bias_testing.benchmark_runner.pull_model", return_value=True)
    def test_cv_screening_dispatch_writes_summary(self, mock_pull, mock_pool, mock_run):
        mock_run.return_value = pd.DataFrame({"score": [80.0, 90.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            run_model_benchmarks("smollm-135m", "cv-screening", tmpdir, n_runs=7)

            kwargs = mock_run.call_args.kwargs
            assert kwargs["model_name"] == "smollm:135m"
            assert kwargs["n_runs"] == 7
            assert kwargs["pool_client"] is mock_pool.return_value

            summary_path = os.path.join(tmpdir, "smollm-135m", "cv-screening", "results.json")
            with open(summary_path) as f:
                summary = json.load(f)
        assert summary["model"] == "smollm-135m"
        assert summary["ollama_tag"] == "smollm:135m"
        assert summary["benchmark"] == "cv-screening"
        assert summary["n_records"] == 2
        assert summary["mean_score"] == 85.0

    @patch("slm_bias_testing.benchmark_runner._get_benchmark")
    @patch("slm_bias_testing.model_clients.OllamaPoolClient")
    @patch("slm_bias_testing.benchmark_runner.pull_model", return_value=True)
    def test_benchmark_dispatch_evaluate_and_summary(self, mock_pull, mock_pool, mock_get_bm):
        bm = MagicMock()
        bm.evaluate.return_value = {
            "benchmark": "winobias",
            "n_examples": 3,
            "overall_accuracy": 61.5,
            "ignored_key": "dropped",
        }
        mock_get_bm.return_value = bm

        with tempfile.TemporaryDirectory() as tmpdir:
            run_model_benchmarks("smollm-135m", "winobias", tmpdir, max_samples=10)

            bm.evaluate.assert_called_once()
            kwargs = bm.evaluate.call_args.kwargs
            assert kwargs["max_samples"] == 10
            assert kwargs["pool_client"] is mock_pool.return_value
            bm.save_results.assert_called_once()

            summary_path = os.path.join(tmpdir, "smollm-135m", "winobias", "results.json")
            with open(summary_path) as f:
                summary = json.load(f)
        assert summary["n_examples"] == 3
        assert summary["overall_accuracy"] == 61.5
        assert summary["max_samples"] == 10
        assert summary["timestamp"]
        assert "ignored_key" not in summary

    @patch("slm_bias_testing.cv_screening.run_cv_screening")
    @patch("slm_bias_testing.model_clients.OllamaPoolClient", side_effect=RuntimeError("no node"))
    @patch("slm_bias_testing.benchmark_runner.pull_model", return_value=True)
    def test_pool_failure_falls_back_to_sequential(self, mock_pull, mock_pool, mock_run):
        mock_run.return_value = pd.DataFrame({"score": [50.0]})
        with tempfile.TemporaryDirectory() as tmpdir:
            run_model_benchmarks("smollm-135m", "cv-screening", tmpdir)
        assert mock_run.call_args.kwargs["pool_client"] is None

    @patch("slm_bias_testing.benchmark_runner.run_model_benchmarks")
    @patch("slm_bias_testing.benchmark_runner.pull_model", return_value=True)
    def test_main_expands_all_models_from_registry(self, mock_pull, mock_run, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--models", "all"])
        main()
        assert mock_run.call_count == len(MODELS)

    @patch("slm_bias_testing.benchmark_runner.run_model_benchmarks")
    def test_main_rejects_unknown_model(self, mock_run, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--models", "bogus-model"])
        main()
        mock_run.assert_not_called()

    @patch("slm_bias_testing.benchmark_runner.run_model_benchmarks")
    def test_main_parses_model_list(self, mock_run, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["run_benchmarks.py", "--models", "smollm-135m, gemma3-270m"]
        )
        main()
        assert [c.args[0] for c in mock_run.call_args_list] == ["smollm-135m", "gemma3-270m"]


class TestSummaryWriting:
    def test_build_benchmark_summary_passthrough(self):
        results = {"n_examples": 5, "overall_accuracy": 61.5, "ignored": "x"}
        out = _build_benchmark_summary("m", "tag", "winobias", results, max_samples=10)
        assert out["n_examples"] == 5
        assert out["overall_accuracy"] == 61.5
        assert out["max_samples"] == 10
        assert out["timestamp"]
        assert "ignored" not in out

    def test_write_summary_atomic_roundtrip(self, tmp_path):
        path = tmp_path / "results.json"
        _write_summary(str(path), {"a": 1})
        assert json.loads(path.read_text()) == {"a": 1}
        assert list(tmp_path.glob("*.tmp")) == []
