"""Tests for the v2 decision instrument (separate from generative prompts)."""

from __future__ import annotations

from slm_bias_testing.data.cvs import cvs
from slm_bias_testing.decision_instrument import (
    BUDGET_TOKENS,
    JD_STUB,
    SCORE_LEVELS,
    decision_state,
    parse_state_tokens,
    score_from_answer,
    score_question,
)


class TestBudgetAndStub:
    def test_budget_constant(self) -> None:
        assert BUDGET_TOKENS == 500

    def test_stub_is_small(self) -> None:
        assert len(JD_STUB.split()) <= 40

    def test_stub_mentions_key_factors(self) -> None:
        # university/degree and A-levels are the manipulated factors
        lower = JD_STUB.lower()
        assert "degree" in lower
        assert "a-level" in lower


class TestDecisionState:
    def test_cv_injected_verbatim(self) -> None:
        state = decision_state(cvs[0])
        assert cvs[0]["cv"] in state

    def test_no_generative_boilerplate(self) -> None:
        state = decision_state(cvs[0])
        for marker in ("XX/100", "Respond with only one line", "score this candidate out of 100"):
            assert marker not in state

    def test_frame_present(self) -> None:
        state = decision_state(cvs[0])
        assert state.startswith("Screening for:")
        assert "Candidate CV" in state


class TestScoreQuestion:
    def test_eleven_levels_0_to_100(self) -> None:
        assert len(SCORE_LEVELS) == 11
        assert SCORE_LEVELS[0] == "0"
        assert SCORE_LEVELS[-1] == "100"
        assert [int(v) for v in SCORE_LEVELS] == list(range(0, 101, 10))

    def test_question_shape(self) -> None:
        question = score_question()["score"]
        assert question["type"] == "score"
        assert question["criteria"] == list(SCORE_LEVELS)
        assert question["instructions"]


class TestParseStateTokens:
    def test_overflow_message(self) -> None:
        msg = "question 0: state has 847 tokens; limit is 477 with this question"
        assert parse_state_tokens(msg) == 847

    def test_no_match(self) -> None:
        assert parse_state_tokens("connection refused") is None


class TestScoreFromAnswer:
    def _answer(self, **overrides: object) -> dict:
        answer: dict = {
            "type": "score",
            "score": 1.52,
            "legend": {"0": "0", "7": "70", "10": "100"},
            "probabilities": {"0": 0.1, "7": 0.6, "3": 0.3},
            "confidence": 0.31,
        }
        answer.update(overrides)
        return answer

    def test_full_mapping(self) -> None:
        out = score_from_answer(self._answer())
        assert out["score_discrete"] == 70  # legend label of argmax index
        assert out["score_continuous"] == 15.2  # 1.52 * 10
        assert out["confidence"] == 0.31
        assert out["probabilities"]["7"] == 0.6

    def test_no_legend_falls_back_to_index(self) -> None:
        out = score_from_answer(self._answer(legend=None))
        assert out["score_discrete"] == 70  # argmax index "7" * 10

    def test_tie_breaks_to_lower_level(self) -> None:
        out = score_from_answer(
            self._answer(probabilities={"2": 0.5, "5": 0.5}, legend={"2": "20", "5": "50"})
        )
        assert out["score_discrete"] == 20

    def test_non_score_type_degrades(self) -> None:
        out = score_from_answer({"type": "noul", "noul": 0.9})
        assert out["score_discrete"] is None
        assert out["score_continuous"] is None
        assert out["probabilities"] == {}

    def test_missing_score_gives_no_continuous(self) -> None:
        out = score_from_answer(self._answer(score=None))
        assert out["score_continuous"] is None

    def test_out_of_range_label_rejected(self) -> None:
        out = score_from_answer(self._answer(probabilities={"4": 1.0}, legend={"4": "999"}))
        assert out["score_discrete"] is None

    def test_non_finite_values_rejected(self) -> None:
        out = score_from_answer(
            self._answer(
                score=float("nan"),
                confidence=float("inf"),
                probabilities={"3": 0.9, "4": float("nan")},
            )
        )
        assert out["score_continuous"] is None
        assert out["confidence"] is None
        assert "4" not in out["probabilities"]
        assert out["score_discrete"] == 30

    def test_non_numeric_probability_ignored(self) -> None:
        out = score_from_answer(self._answer(probabilities={"7": "garbage", "3": 0.9}))
        assert out["score_discrete"] == 30
