from unittest.mock import patch

from slm_bias_testing.benchmark_runner import run_model_benchmarks


class TestRunModelBenchmarks:
    @patch("slm_bias_testing.model_clients.OllamaPoolClient")
    @patch("slm_bias_testing.benchmark_runner.pull_model", return_value=True)
    @patch("slm_bias_testing.benchmark_runner.get_model")
    def test_skip_existing_results(self, mock_get_model, mock_pull, mock_pool):
        import json
        import os
        import tempfile

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
        import os
        import tempfile

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
