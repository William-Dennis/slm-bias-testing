"""Tests for OllamaPoolClient."""

from __future__ import annotations

import json
from typing import cast
from unittest.mock import MagicMock, patch

from slm_bias_testing.call_api import DEFAULT_NUM_PREDICT
from slm_bias_testing.model_clients import OllamaPoolClient, PoolClientProtocol, SequentialPredictor


def _mock_popen_with_ready():
    """Create a mock Popen that emits the JSON handshake then results."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    # Pre-configure stdout so readline is stable across accesses
    stdout_mock = MagicMock()
    stdout_mock.readline.side_effect = [
        json.dumps({"protocol": 1, "ready": True}) + "\n",
    ]
    proc.stdout = stdout_mock
    proc.stderr = iter([])
    proc.wait.return_value = 0
    return proc


class TestOllamaPoolClient:
    @patch("slm_bias_testing.model_clients.subprocess.Popen")
    def test_init_spawns_pool(self, mock_popen):
        mock_popen.return_value = _mock_popen_with_ready()
        client = OllamaPoolClient(model_name="smollm:135m", pool_size=4)
        cmd = mock_popen.call_args[0][0]
        assert "node" in cmd[0]
        assert "ollama_pool.mjs" in cmd[1]
        client.close()

    @patch("slm_bias_testing.model_clients.subprocess.Popen")
    def test_predict_batch_writes_and_reads(self, mock_popen):
        proc = _mock_popen_with_ready()
        handshake = json.dumps({"protocol": 1, "ready": True}) + "\n"
        proc.stdout.readline.side_effect = [
            handshake,  # consumed during __init__
            json.dumps({"id": "j1", "response": "hi", "error": None, "latency_ms": 100}) + "\n",
            json.dumps({"id": "j2", "response": "yo", "error": None, "latency_ms": 200}) + "\n",
        ]
        mock_popen.return_value = proc
        client = OllamaPoolClient(model_name="test-model")
        results = client.predict_batch(
            [
                {"id": "j1", "prompt": "a"},
                {"id": "j2", "prompt": "b"},
            ]
        )
        assert results["j1"]["response"] == "hi"
        assert results["j2"]["response"] == "yo"
        client.close()

    @patch("slm_bias_testing.model_clients.subprocess.Popen")
    def test_predict_batch_empty(self, mock_popen):
        mock_popen.return_value = _mock_popen_with_ready()
        client = OllamaPoolClient(model_name="test-model")
        assert client.predict_batch([]) == {}
        client.close()

    @patch("slm_bias_testing.model_clients.subprocess.Popen")
    def test_close_terminates(self, mock_popen):
        mock_popen.return_value = _mock_popen_with_ready()
        client = OllamaPoolClient(model_name="test-model")
        client.close()
        wait_called = cast("MagicMock", client._proc.wait).called
        assert wait_called

    @patch("slm_bias_testing.model_clients.subprocess.Popen")
    def test_context_manager(self, mock_popen):
        mock_popen.return_value = _mock_popen_with_ready()
        with OllamaPoolClient(model_name="test-model") as c:
            assert c is not None
        wait_called = cast("MagicMock", c._proc.wait).called
        assert wait_called

    @patch("slm_bias_testing.model_clients.subprocess.Popen")
    def test_adaptive_default(self, mock_popen):
        """Default is adaptive=True — no --no-adaptive flag."""
        mock_popen.return_value = _mock_popen_with_ready()
        client = OllamaPoolClient(model_name="test-model")
        cmd = mock_popen.call_args[0][0]
        assert "--no-adaptive" not in cmd
        client.close()

    @patch("slm_bias_testing.model_clients.subprocess.Popen")
    def test_no_adaptive_explicit(self, mock_popen):
        mock_popen.return_value = _mock_popen_with_ready()
        client = OllamaPoolClient(model_name="test-model", adaptive=False)
        cmd = mock_popen.call_args[0][0]
        assert "--no-adaptive" in cmd
        client.close()

    @patch("slm_bias_testing.model_clients.subprocess.Popen")
    def test_adaptive_explicit(self, mock_popen):
        mock_popen.return_value = _mock_popen_with_ready()
        client = OllamaPoolClient(model_name="test-model", adaptive=True)
        cmd = mock_popen.call_args[0][0]
        assert "--no-adaptive" not in cmd
        client.close()

    @patch("slm_bias_testing.model_clients.subprocess.Popen")
    def test_predict_batch_sends_num_predict(self, mock_popen):
        proc = _mock_popen_with_ready()
        proc.stdout.readline.side_effect = [
            json.dumps({"protocol": 1, "ready": True}) + "\n",
            json.dumps({"id": "j1", "response": "hi", "error": None, "latency_ms": 100}) + "\n",
        ]
        mock_popen.return_value = proc
        client = OllamaPoolClient(model_name="test-model")
        client.predict_batch([{"id": "j1", "prompt": "a"}])
        written = [
            json.loads(c.args[0]) for c in cast("MagicMock", proc.stdin).write.call_args_list
        ]
        assert written[0]["num_predict"] == DEFAULT_NUM_PREDICT
        client.close()

    @patch("slm_bias_testing.model_clients.subprocess.Popen")
    def test_predict_batch_job_num_predict_override(self, mock_popen):
        proc = _mock_popen_with_ready()
        proc.stdout.readline.side_effect = [
            json.dumps({"protocol": 1, "ready": True}) + "\n",
            json.dumps({"id": "j1", "response": "hi", "error": None, "latency_ms": 100}) + "\n",
        ]
        mock_popen.return_value = proc
        client = OllamaPoolClient(model_name="test-model")
        client.predict_batch([{"id": "j1", "prompt": "a", "num_predict": 128}])
        written = [
            json.loads(c.args[0]) for c in cast("MagicMock", proc.stdin).write.call_args_list
        ]
        assert written[0]["num_predict"] == 128
        client.close()

    @patch("slm_bias_testing.model_clients.subprocess.Popen")
    def test_batch_timeout_default(self, mock_popen):
        mock_popen.return_value = _mock_popen_with_ready()
        client = OllamaPoolClient(model_name="test-model")
        assert client.batch_timeout == 300
        client.close()


class TestSequentialPredictor:
    """SequentialPredictor adapts Model.predict to the pool protocol."""

    class _StubModel:
        def __init__(self, fail: bool = False):
            self.fail = fail
            self.calls: list[tuple[str, float, int | None]] = []

        def predict(
            self, prompt: str, temperature: float = 0.0, num_predict: int | None = None
        ) -> str:
            self.calls.append((prompt, temperature, num_predict))
            if self.fail:
                raise RuntimeError("ollama down")
            return "85/100"

    def test_predict_batch_success(self):
        model = self._StubModel()
        client = SequentialPredictor(model)
        results = client.predict_batch([{"id": "j1", "prompt": "hi", "temperature": 0.5}])
        assert results == {"j1": {"response": "85/100", "error": None}}
        assert model.calls == [("hi", 0.5, None)]

    def test_predict_batch_forwards_num_predict(self):
        model = self._StubModel()
        client = SequentialPredictor(model)
        client.predict_batch([{"id": "j1", "prompt": "hi", "num_predict": 128}])
        assert model.calls == [("hi", 0.0, 128)]

    def test_predict_batch_captures_error(self):
        client = SequentialPredictor(self._StubModel(fail=True))
        results = client.predict_batch([{"id": "j1", "prompt": "hi"}])
        assert results["j1"]["response"] is None
        assert "ollama down" in results["j1"]["error"]

    def test_satisfies_pool_protocol(self):
        def takes_pool(client: PoolClientProtocol) -> int:
            return client.batch_size

        client = SequentialPredictor(self._StubModel())
        assert takes_pool(client) == 1
        assert client.close() is None
