from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    name: str = "base"

    def _checkpoint_path(self, output_dir: str) -> str:
        """Path to the JSONL checkpoint file for incremental saves."""
        return os.path.join(output_dir, f"{self.name}_checkpoint.jsonl")

    def _save_call(self, output_dir: str, call_data: dict[str, Any]) -> None:
        """Append one LLM call result to the checkpoint file.

        Called after every model.predict() so that a process kill
        loses at most the in-flight call, not hours of work.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = self._checkpoint_path(output_dir)
        with open(path, "a") as f:
            f.write(json.dumps(call_data) + "\n")
            f.flush()

    def _load_checkpoint(self, output_dir: str) -> list[dict[str, Any]]:
        """Load all checkpointed call results (JSONL, one dict per line)."""
        path = self._checkpoint_path(output_dir)
        if not os.path.exists(path):
            return []
        results: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Truncated checkpoint line — stopping at last valid entry")
                    break
        return results

    @abstractmethod
    def load_dataset(self) -> list[Any]: ...

    @abstractmethod
    def evaluate(
        self, model: Any, max_samples: int | None = None, output_dir: str | None = None
    ) -> dict[str, Any]: ...

    def save_results(self, results: dict[str, Any], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"{self.name}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    def load_results(self, output_dir: str) -> dict[str, Any] | None:
        results_file = os.path.join(output_dir, f"{self.name}.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                return json.load(f)  # type: ignore[no-any-return]
        return None
