"""Ollama-backed model client with automatic server recovery."""

from __future__ import annotations

import contextlib
import logging
import os
import time

import ollama

from slm_bias_testing.ollama_setup import OllamaServer

logger = logging.getLogger(__name__)

LLM_MODEL = "gemma3:1b-it-qat"
PROVIDER = "ollama"

# Default context window. Ollama 0.19+ uses MLX on Apple Silicon with intelligent
# KV cache checkpoints — setting an explicit num_ctx lets Ollama optimise cache
# allocation.  See: https://ollama.com/blog/mlx
DEFAULT_NUM_CTX = max(256, min(131072, int(os.environ.get("SLM_NUM_CTX", "2048"))))

# How long to keep the model loaded after last API call (seconds).
# Longer values reduce model reload overhead across sequential benchmark items.
DEFAULT_KEEP_ALIVE = max(0.0, min(300.0, float(os.environ.get("SLM_KEEP_ALIVE", "5"))))


class OllamaClient:
    """Thin wrapper around an Ollama client with auto-restart on failure."""

    def __init__(self, timeout: int = 300) -> None:
        self._client = ollama.Client(timeout=timeout)
        self._server: OllamaServer | None = None

    def ensure_running(self) -> None:
        """Check if Ollama is responding; restart if not."""
        try:
            self._client.list()
            return
        except Exception:
            logger.warning("Ollama not responding, restarting...")
            if self._server is not None:
                with contextlib.suppress(Exception):
                    self._server.stop()
            try:
                self._server = OllamaServer(kill_existing=True)
                self._server.start()
            except Exception:
                logger.error(
                    "Ollama auto-restart failed — subsequent API calls will fail", exc_info=True
                )
                raise
            logger.info("Ollama restarted")

    @property
    def client(self) -> ollama.Client:
        return self._client


class Model:
    """Ollama prediction interface with retry and auto-restart."""

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        provider: str = PROVIDER,
        ollama_client: OllamaClient | None = None,
        num_ctx: int | None = None,
        keep_alive: float | None = None,
    ) -> None:
        if provider != "ollama":
            raise ValueError(f"Only 'ollama' provider supported, got '{provider}'")
        self.model_name = model_name
        self.num_ctx = num_ctx if num_ctx is not None else DEFAULT_NUM_CTX
        self.keep_alive = keep_alive if keep_alive is not None else DEFAULT_KEEP_ALIVE
        self._ollama_client = ollama_client or OllamaClient()
        self._ollama_client.ensure_running()

    def predict(self, input_text: str, temperature: float = 0.0) -> str:
        """Run a single prediction via Ollama."""
        max_retries = 3
        start = time.monotonic()
        for attempt in range(max_retries):
            try:
                response = self._ollama_client.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": input_text}],
                    options={
                        "temperature": temperature,
                        "num_ctx": self.num_ctx,
                    },
                    keep_alive=self.keep_alive,
                )
                elapsed = time.monotonic() - start
                logger.debug("Ollama call completed in %.2fs (attempt %d)", elapsed, attempt + 1)
                return response["message"]["content"]  # type: ignore[no-any-return]
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Ollama call failed (attempt %d/%d): %s", attempt + 1, max_retries, e
                    )
                    # Only restart on actual connection failure, not on every error
                    if "connect" in str(e).lower() or "refused" in str(e).lower():
                        self._ollama_client.ensure_running()
                    sleep_time = min(2 ** (attempt + 1), 30)
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        "Ollama call failed after %d attempts: %s", max_retries, e, exc_info=True
                    )
                    raise
        raise RuntimeError("Unreachable")
