from __future__ import annotations

import atexit
import logging
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


def ollama_alive(timeout: float = 2.0) -> bool:
    """True when a local Ollama server answers ``/api/tags``.

    The single Python-side liveness probe — callers (client recovery,
    server wait loops, scripts) must use this instead of re-implementing
    the HTTP check. The Node.js pool keeps its own probe (separate process,
    no Python import possible).
    """
    host = os.environ.get("OLLAMA_HOST", "localhost:11434")
    base = host if "://" in host else f"http://{host}"
    try:
        with urllib.request.urlopen(f"{base.rstrip('/')}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


class OllamaServer:
    def __init__(self, kill_existing: bool = True):
        self.process: subprocess.Popen[bytes] | None = None
        if kill_existing:
            self._kill_existing_ollama()

    def _kill_existing_ollama(self) -> None:
        # Windows
        if platform.system() == "Windows":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ollama.exe"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                logger.exception("Failed to kill existing ollama process on Windows")
        # macOS / Linux
        else:
            try:
                subprocess.run(
                    ["pkill", "-f", "ollama serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                logger.exception("Failed to kill existing ollama process")

    def start(self) -> None:
        if self.process is not None:
            return  # already running

        self.process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_server()
        atexit.register(self.stop)

    def _wait_for_server(self, timeout: int = 30, interval: int = 1) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if ollama_alive(timeout=2):
                logger.info("Ollama server is ready")
                return
            time.sleep(interval)
        # Timeout — kill the process
        if self.process:
            try:
                self.process.kill()
            except Exception:
                logger.exception("Failed to kill ollama server process")
            self.process = None
        raise RuntimeError(f"Ollama server did not start within {timeout} seconds")

    def stop(self) -> None:
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
