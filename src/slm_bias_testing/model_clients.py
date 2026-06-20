"""Ollama pool client — communicates with the Node.js worker pool."""

from __future__ import annotations

import contextlib
import json
import logging
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)


class BatchTimeoutError(Exception):
    """Raised when a predict_batch call exceeds its configured timeout."""


class OllamaPoolClient:
    """Client that communicates with ollama_pool.mjs subprocess."""

    def __init__(
        self,
        model_name: str,
        pool_size: int = 4,
        batch_size: int = 40,
        adaptive: bool = True,
        batch_timeout: int = 300,
        ollama_host: str = "http://localhost:11434",
        ram_floor: float = 2.0,
        latency_ceiling: int = 15000,
        timeout: int = 30000,
        retries: int = 2,
        keep_alive: float = 30.0,
        num_ctx: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx

        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "ollama_pool.mjs"
        cmd = [
            "node",
            str(script_path),
            "--max-pool",
            str(pool_size),
            "--ram-floor",
            str(ram_floor),
            "--latency-ceiling",
            str(latency_ceiling),
            "--timeout",
            str(timeout),
            "--retries",
            str(retries),
            "--ollama-host",
            ollama_host,
        ]
        if not adaptive:
            cmd.append("--no-adaptive")

        logger.info("Starting pool: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._read_lock = threading.Lock()
        self._stderr_lines: deque[str] = deque(maxlen=100)
        self._ready_event = threading.Event()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        # Wait for pool to signal readiness (drain thread sets the event)
        if not self._ready_event.wait(timeout=30):
            self.close()
            raise RuntimeError("Ollama pool failed to start within 30s")

    def predict_batch(self, jobs: list[dict]) -> dict[str, dict]:
        """Send batch of jobs to pool, read results as they stream back.

        Args:
            jobs: [{"id": "cv_001", "prompt": "...", "temperature": 0.0}, ...]

        Returns:
            {"cv_001": {"response": "85/100", "error": None}, ...}
        """
        if not jobs:
            return {}

        # Build full job dicts with model params
        full_jobs = []
        for job in jobs:
            full_jobs.append(
                {
                    "id": job["id"],
                    "model": self.model_name,
                    "prompt": job["prompt"],
                    "temperature": job.get("temperature", 0.0),
                    "num_ctx": self.num_ctx,
                    "keep_alive": self.keep_alive,
                }
            )

        # Write all jobs to pool stdin
        if self._proc.stdin is None:
            raise RuntimeError("Pool subprocess stdin is closed")
        for job in full_jobs:
            self._proc.stdin.write(json.dumps(job) + "\n")
        self._proc.stdin.flush()

        # Read exactly len(jobs) results from stdout with a batch-level timeout
        results: dict[str, dict] = {}
        expected_ids = {job["id"] for job in full_jobs}
        old_alarm = signal.alarm(0)  # Save and disarm any existing alarm
        signal.signal(signal.SIGALRM, self._batch_timeout_handler)
        signal.alarm(self.batch_timeout)
        try:
            with self._read_lock:
                for _ in range(len(full_jobs)):
                    line = self._read_line()
                    if line is None:
                        logger.error("Pool subprocess closed before all results received")
                        break
                    try:
                        parsed = json.loads(line)
                        job_id = parsed["id"]
                        results[job_id] = {
                            "response": parsed.get("response"),
                            "error": parsed.get("error"),
                        }
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error("Failed to parse pool result: %s", e)
        except BatchTimeoutError:
            logger.error(
                "Batch timed out after %ds — got %d/%d results",
                self.batch_timeout,
                len(results),
                len(full_jobs),
            )
            raise
        finally:
            signal.alarm(old_alarm)  # Restore previous alarm

        missing = expected_ids - results.keys()
        if missing:
            raise RuntimeError(
                f"Pool returned incomplete results — missing {len(missing)} job(s): {missing!s}"
            )

        return results

    @staticmethod
    def _batch_timeout_handler(signum: int, frame: object | None) -> None:
        raise BatchTimeoutError("Batch read exceeded configured timeout")

    def _read_line(self) -> str | None:
        """Read a single line from pool stdout. Returns None on EOF."""
        if self._proc.stdout is None:
            raise RuntimeError("Pool subprocess stdout is closed")
        return self._proc.stdout.readline() or None

    def _drain_stderr(self) -> None:
        """Background thread: read pool stderr to prevent buffer deadlock.

        Also detects the Ollama ready signal and sets ``_ready_event``.
        """
        if self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            stripped = line.rstrip("\n")
            self._stderr_lines.append(stripped)
            if not self._ready_event.is_set() and (
                "Ollama started" in stripped or "Ollama already running" in stripped
            ):
                self._ready_event.set()

    def close(self) -> None:
        """Close stdin and wait for pool to drain and exit."""
        if self._proc.stdin:
            with contextlib.suppress(OSError):
                self._proc.stdin.close()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("Pool did not exit in time, terminating")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Pool did not terminate, killing")
                self._proc.kill()

        self._stderr_thread.join(timeout=5)

        # Log stderr for diagnostics
        for line in self._stderr_lines:
            logger.debug("pool: %s", line)

    def __enter__(self) -> OllamaPoolClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
