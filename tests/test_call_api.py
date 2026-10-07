"""Tests for slm_bias_testing.call_api — Ollama and model clients."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from slm_bias_testing.call_api import (
    DEFAULT_KEEP_ALIVE,
    DEFAULT_NUM_CTX,
    LLM_MODEL,
    Model,
    OllamaClient,
)


class TestOllamaClient:
    def test_ensure_running_success(self):
        client = OllamaClient.__new__(OllamaClient)
        client._client = MagicMock()
        client._server = None
        client.ensure_running()
        client._client.list.assert_called_once()

    def test_ensure_running_restarts_on_failure(self):
        client = OllamaClient.__new__(OllamaClient)
        client._client = MagicMock()
        client._client.list.side_effect = ConnectionError("down")
        client._server = MagicMock()

        with patch("slm_bias_testing.call_api.OllamaServer") as MockServer:
            mock_instance = MagicMock()
            MockServer.return_value = mock_instance
            client.ensure_running()
            mock_instance.start.assert_called_once()


class TestModelInit:
    def test_model_init_default(self):
        with patch.object(OllamaClient, "ensure_running"):
            model = Model()
        assert model.model_name == LLM_MODEL
        assert model.num_ctx == DEFAULT_NUM_CTX
        assert model.keep_alive == DEFAULT_KEEP_ALIVE

    def test_model_init_custom(self):
        with patch.object(OllamaClient, "ensure_running"):
            model = Model(model_name="custom-model", num_ctx=4096, keep_alive=10.0)
        assert model.model_name == "custom-model"
        assert model.num_ctx == 4096
        assert model.keep_alive == 10.0

    def test_model_init_with_custom_client(self):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.ensure_running = MagicMock()
        model = Model(model_name="test", provider="ollama", ollama_client=mock_client)
        mock_client.ensure_running.assert_called_once()
        assert model._ollama_client is mock_client


class TestModelPredict:
    def test_predict_unknown_provider(self):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.ensure_running = MagicMock()
        with pytest.raises(ValueError, match="Only 'ollama' provider supported"):
            Model(model_name="test", provider="unknown", ollama_client=mock_client)

    def test_predict_ollama_success(self):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.ensure_running = MagicMock()
        mock_client.client = MagicMock()
        mock_client.client.chat.return_value = {"message": {"content": "42/100"}}
        model = Model(model_name="test", provider="ollama", ollama_client=mock_client)
        result = model.predict("score this", temperature=0.5)
        assert result == "42/100"
        # Verify num_ctx and keep_alive are passed through
        call_kwargs = mock_client.client.chat.call_args
        assert call_kwargs.kwargs["options"]["num_ctx"] == DEFAULT_NUM_CTX
        assert call_kwargs.kwargs["keep_alive"] == DEFAULT_KEEP_ALIVE
        assert result == "42/100"

    def test_predict_ollama_retries_on_failure(self):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.ensure_running = MagicMock()
        mock_client.client = MagicMock()
        mock_client.client.chat.side_effect = [
            ConnectionError("connection refused"),
            {"message": {"content": "ok"}},
        ]
        model = Model(model_name="test", provider="ollama", ollama_client=mock_client)
        result = model.predict("hello")
        assert result == "ok"
        assert mock_client.client.chat.call_count == 2
        # Connection error should trigger ensure_running restart
        assert mock_client.ensure_running.call_count >= 1
        assert mock_client.client.chat.call_count == 2
