"""Filesystem helpers shared by benchmarks and runners."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from typing import Any


def atomic_write_text(path: str, text: str) -> None:
    """Write text to path atomically.

    Uses a unique temporary sibling per call (safe when several writers
    target the same path) and removes it if the write fails.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_json(path: str, payload: Any) -> None:
    """Atomically write payload as pretty-printed JSON.

    A crash never leaves a truncated results file behind.
    """
    atomic_write_text(path, json.dumps(payload, indent=2))
