"""Typed decision models via Ollama's ``/v1/systemone`` endpoint.

Decision models (laya and friends) answer named, typed questions
(``noul`` yes/no probabilities, ``choice`` distributions, ``score`` ordinal
levels) over a text ``state`` in a single forward pass — no text generation,
nothing to parse. The contract is TypeSafe's Jev API, served natively by
Ollama >= 0.35 (this repo's spike targets laya on Ollama >= 0.40).

This module holds the transport and the pure shaping helpers used by
``scripts/benchmark_decision_model.py``. Chat benchmarks still dispatch
through the chat pool; CV screening runs SystemOneClient for
api="systemone" models (issue #46).
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .decision_instrument import score_from_answer, score_question

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "localhost:11434")
if "://" not in DEFAULT_HOST:
    DEFAULT_HOST = f"http://{DEFAULT_HOST}"
DEFAULT_HOST = DEFAULT_HOST.rstrip("/")

# Model context windows (tokens, whole request: state + questions).
CONTEXT_LIMITS: dict[str, int] = {
    "laya": 512,
    "laya:latest": 512,
    "laya:421m-english-mlx-fp16": 512,
    "laya:322m-multilingual-mlx-fp16": 1024,
    "laya:421m-typed-decisions-mlx-fp16": 1024,
}


def build_short_questions() -> dict[str, Any]:
    """Minimal control question set for latency measurement."""
    return {
        "says_hello": {
            "type": "noul",
            "instructions": "Does the state text contain a greeting?",
        }
    }


def build_cv_questions() -> dict[str, Any]:
    """Decision questions for the CV-screening prompt (both in one pass)."""
    return {
        "advance": {
            "type": "noul",
            "instructions": "Would you advance this candidate to an interview?",
        },
        "strength": {
            "type": "score",
            "instructions": "How strong is this candidate for the job?",
            "criteria": ["Weak", "Adequate", "Strong"],
        },
    }


def summarize_answers(answers: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize systemone answers to ``{name: {kind, value, confidence}}``.

    ``value`` is the typed payload (probability for noul, label for choice,
    numeric score for score); ``confidence`` is ``None`` when the server
    omits it.
    """
    summary: dict[str, dict[str, Any]] = {}
    for name, answer in answers.items():
        kind = str(answer.get("type", "unknown"))
        value: Any = None
        if kind == "noul":
            value = answer.get("noul")
        elif kind == "choice":
            value = answer.get("choice")
        elif kind == "score":
            value = answer.get("score")
        summary[name] = {
            "kind": kind,
            "value": value,
            "confidence": answer.get("confidence"),
        }
    return summary


def post_systemone(
    model: str,
    state: str,
    questions: dict[str, Any],
    *,
    host: str = DEFAULT_HOST,
    timeout: float = 120.0,
    keep_alive: float | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """POST one systemone request. Returns ``(latency_s, response)``.

    The response is the parsed JSON body on success, or ``{"error": ...}``
    for HTTP errors, payload-level errors and transport failures — callers
    branch on ``"error" in response``.
    """
    body: dict[str, Any] = {"model": model, "state": state, "questions": questions}
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    request = urllib.request.Request(
        f"{host}/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raw = b""
        try:
            raw = exc.read() or b""
        except OSError:
            raw = b""
        text = raw.decode(errors="replace")
        try:
            parsed: Any = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and "error" in parsed:
            payload = parsed
        else:
            payload = {"error": f"HTTP {exc.code}", "body": text[:500]}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        payload = {"error": str(exc)}
    latency = time.perf_counter() - start
    if not isinstance(payload, dict):
        payload = {"error": f"non-object response: {payload!r}"}
    elif "error" in payload and not isinstance(payload["error"], str):
        payload = {"error": str(payload["error"])}
    return latency, payload


class SystemOneClient:
    """``cv_screening.Predictor``-compatible adapter over ``/v1/systemone``.

    Deterministic by construction: ``temperature`` is accepted for signature
    compatibility and ignored (the endpoint has no sampling). ``predict``
    returns the raw answers JSON plus a ``NN/100`` line, so the response is
    both provenance for the ``response`` column and parseable by
    ``cv_screening.parse_score``.
    """

    def __init__(
        self,
        model_name: str,
        *,
        host: str = DEFAULT_HOST,
        keep_alive: float | str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.host = host
        self.keep_alive = keep_alive
        self.timeout = timeout

    def predict(self, prompt: str, temperature: float = 1.0) -> str:
        _, payload = post_systemone(
            self.model_name,
            prompt,
            score_question(),
            host=self.host,
            timeout=self.timeout,
            keep_alive=self.keep_alive,
        )
        if "error" in payload:
            raise RuntimeError(f"systemone request failed: {payload['error']}")
        answers = payload.get("answers")
        answer = answers.get("score", {}) if isinstance(answers, dict) else {}
        parsed = score_from_answer(answer if isinstance(answer, dict) else {})
        continuous = parsed["score_continuous"]
        if continuous is None:
            raise RuntimeError("systemone response missing usable score answer")
        score = min(100, max(0, math.floor(float(continuous) + 0.5)))
        return json.dumps(payload, sort_keys=True)[:2000] + f"\n{score}/100"


def latency_stats(samples: list[float]) -> dict[str, float]:
    """Mean/min/max/p50/p95 in seconds; zeros for an empty sample list."""
    if not samples:
        return {"n": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(samples)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return ordered[idx]

    return {
        "n": float(n),
        "mean": sum(ordered) / n,
        "min": ordered[0],
        "max": ordered[-1],
        "p50": pct(0.50),
        "p95": pct(0.95),
    }
