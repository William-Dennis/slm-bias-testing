"""Filesystem helpers shared by benchmarks and runners."""

from __future__ import annotations

import json
import os
from typing import Any


def atomic_write_text(path: str, text: str) -> None:
    """Write text to path atomically (tmp file + os.replace)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_json(path: str, payload: Any) -> None:
    """Atomically write payload as pretty-printed JSON.

    A crash never leaves a truncated results file behind.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
