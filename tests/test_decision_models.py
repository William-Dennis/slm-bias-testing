"""Tests for decision_models: shaping helpers, stats, systemone transport."""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import TracebackType

from slm_bias_testing.decision_models import (
    CONTEXT_LIMITS,
    build_cv_questions,
    build_short_questions,
    latency_stats,
    post_systemone,
    summarize_answers,
)


class TestQuestions:
    def test_short_is_single_noul(self) -> None:
        questions = build_short_questions()
        assert list(questions) == ["says_hello"]
        assert questions["says_hello"]["type"] == "noul"
        assert questions["says_hello"]["instructions"]

    def test_cv_has_noul_and_score(self) -> None:
        questions = build_cv_questions()
        assert questions["advance"]["type"] == "noul"
        strength = questions["strength"]
        assert strength["type"] == "score"
        assert strength["criteria"] == ["Weak", "Adequate", "Strong"]

    def test_context_limits_cover_laya_tags(self) -> None:
        for tag in (
            "laya",
            "laya:322m-multilingual-mlx-fp16",
            "laya:421m-typed-decisions-mlx-fp16",
        ):
            assert tag in CONTEXT_LIMITS
            assert CONTEXT_LIMITS[tag] in (512, 1024)


class TestSummarizeAnswers:
    def test_noul(self) -> None:
        out = summarize_answers({"a": {"type": "noul", "noul": 0.9}})
        assert out == {"a": {"kind": "noul", "value": 0.9, "confidence": None}}

    def test_choice(self) -> None:
        out = summarize_answers({"q": {"type": "choice", "choice": "billing"}})
        assert out["q"]["kind"] == "choice"
        assert out["q"]["value"] == "billing"

    def test_score_keeps_confidence(self) -> None:
        out = summarize_answers({"q": {"type": "score", "score": 1.5, "confidence": 0.3}})
        assert out["q"]["value"] == 1.5
        assert out["q"]["confidence"] == 0.3

    def test_unknown_type(self) -> None:
        out = summarize_answers({"q": {"weird": True}})
        assert out["q"] == {"kind": "unknown", "value": None, "confidence": None}


class TestLatencyStats:
    def test_empty(self) -> None:
        stats = latency_stats([])
        assert stats["n"] == 0.0
        assert stats["p50"] == 0.0
        assert stats["p95"] == 0.0

    def test_known_values(self) -> None:
        stats = latency_stats([4.0, 1.0, 3.0, 2.0])
        assert stats["n"] == 4.0
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0
        assert stats["p50"] == 3.0  # nearest-rank on sorted [1,2,3,4]
        assert stats["p95"] == 4.0

    def test_single(self) -> None:
        stats = latency_stats([0.5])
        assert stats["p50"] == 0.5
        assert stats["p95"] == 0.5


class TestPostSystemone:
    def _patch_urlopen(self, monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
        class FakeResponse(io.BytesIO):
            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> None:
                return None

        def fake_urlopen(request: object, timeout: float = 0) -> FakeResponse:
            return FakeResponse(json.dumps(payload).encode())

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen, raising=False)

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_urlopen(monkeypatch, {"answers": {"q": {"type": "noul", "noul": 0.5}}})
        latency, payload = post_systemone("laya", "hi", {"q": {"type": "noul"}})
        assert latency >= 0
        assert "error" not in payload
        assert payload["answers"]["q"]["noul"] == 0.5

    def test_http_error_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = json.dumps({"error": "state has 847 tokens; limit is 477"}).encode()

        def fake_urlopen(request: object, timeout: float = 0) -> None:
            raise urllib.error.HTTPError(
                "http://x", 400, "Bad Request", Message(), io.BytesIO(body)
            )

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen, raising=False)
        _, payload = post_systemone("laya", "hi", {})
        assert "847 tokens" in str(payload["error"])

    def test_http_error_body_without_error_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = json.dumps({"detail": "invalid request"}).encode()

        def fake_urlopen(request: object, timeout: float = 0) -> None:
            raise urllib.error.HTTPError(
                "http://x", 400, "Bad Request", Message(), io.BytesIO(body)
            )

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen, raising=False)
        _, payload = post_systemone("laya", "hi", {})
        assert "HTTP 400" in str(payload["error"])

    def test_non_object_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_urlopen_raw(monkeypatch, b"null")
        _, payload = post_systemone("laya", "hi", {})
        assert "non-object response" in str(payload["error"])

    def _patch_urlopen_raw(self, monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
        class FakeResponse(io.BytesIO):
            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> None:
                return None

        def fake_urlopen(request: object, timeout: float = 0) -> FakeResponse:
            return FakeResponse(raw)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen, raising=False)

    def test_connection_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(request: object, timeout: float = 0) -> None:
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen, raising=False)
        _, payload = post_systemone("laya", "hi", {})
        assert "error" in payload
