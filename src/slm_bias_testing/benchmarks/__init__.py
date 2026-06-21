from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from slm_bias_testing.model_clients import PoolClientProtocol

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    name: str = "base"

    def _checkpoint_path(self, output_dir: str) -> str:
        return os.path.join(output_dir, f"{self.name}_checkpoint.jsonl")

    def _save_call(self, output_dir: str, call_data: dict[str, Any]) -> None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = self._checkpoint_path(output_dir)
            with open(path, "a") as f:
                f.write(json.dumps(call_data) + "\n")
                f.flush()
        except OSError as e:
            logger.error("Failed to write checkpoint: %s", e)

    def _load_checkpoint(self, output_dir: str) -> list[dict[str, Any]]:
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
                    logger.warning("Skipping corrupt checkpoint line")
        return results

    @staticmethod
    def _process_batch(
        pending_items: list,
        pool_client: PoolClientProtocol,
        build_jobs: Callable[[list], list[dict]],
        process_results: Callable[[list, list[dict], dict[str, dict]], None],
    ) -> None:
        batch_size = pool_client.batch_size
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        for batch_start in range(0, len(pending_items), batch_size):
            batch = pending_items[batch_start : batch_start + batch_size]
            jobs = build_jobs(batch)
            results = pool_client.predict_batch(jobs)
            process_results(batch, jobs, results)

    @abstractmethod
    def load_dataset(self) -> list[Any]: ...

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        max_samples: int | None = None,
        output_dir: str | None = None,
        pool_client: PoolClientProtocol | None = None,
    ) -> dict[str, Any]: ...

    def save_results(self, results: dict[str, Any], output_dir: str) -> None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            results_file = os.path.join(output_dir, f"{self.name}.json")
            tmp_file = results_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump(results, f, indent=2)
            os.replace(tmp_file, results_file)
            ckpt = self._checkpoint_path(output_dir)
            if os.path.exists(ckpt):
                os.remove(ckpt)
        except OSError as e:
            logger.warning("Failed to save results to %s: %s", output_dir, e)

    def load_results(self, output_dir: str) -> dict[str, Any] | None:
        results_file = os.path.join(output_dir, f"{self.name}.json")
        if os.path.exists(results_file):
            try:
                with open(results_file) as f:
                    return json.load(f)  # type: ignore[no-any-return]
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load results from %s: %s", results_file, e)
        return None
