"""Tests for OllamaPoolClient."""

from __future__ import annotations

import json
from typing import cast
from unittest.mock import MagicMock, patch

from slm_bias_testing.model_clients import OllamaPoolClient


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
    def test_batch_timeout_default(self, mock_popen):
        mock_popen.return_value = _mock_popen_with_ready()
        client = OllamaPoolClient(model_name="test-model")
        assert client.batch_timeout == 300
        client.close()
