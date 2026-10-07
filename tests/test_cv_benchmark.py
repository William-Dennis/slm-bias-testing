"""Tests for the CV screening benchmark: scoring, stratification, outputs."""

import json
import re

import pytest

from slm_bias_testing.cv_screening import (
    STATUS_API_ERROR,
    STATUS_OK,
    STATUS_PARSE_ERROR,
    STATUS_SKIPPED,
    build_base_prompt,
    cv_prompt,
    parse_score,
    process_cv_run,
    run_cv_screening,
    sha256_hash,
    stratified_sample,
)
from slm_bias_testing.decision_instrument import decision_base_frame, decision_state

JOB_DESC = "Junior Data Analyst"


def make_cv(index: int, gender: str, ethnicity: str, prestige: str, quality: str, template: str):
    metadata = {
        "name": f"Test Person {index}",
        "name_gender": gender,
        "name_ethnicity": ethnicity,
        "university": f"Uni {prestige}",
        "university_prestige": prestige,
        "a_levels": f"Grades {quality}",
        "a_level_quality": quality,
        "template_name": template,
    }
    return {"cv": f"CV body ID={index} gender={gender}", "metadata": metadata}


def default_responder(prompt: str) -> str:
    """Deterministic, CV-varying score so groups have non-zero variance."""
    match = re.search(r"ID=(\d+)", prompt)
    index = int(match.group(1)) if match else 0
    return f"{70 + (index * 7) % 31}/100"


@pytest.fixture
def small_cvs():
    """32-CV factorial: 2 genders x 2 ethnicities x 2 prestige x 2 quality x 2 templates."""
    return [
        make_cv(i, g, e, p, q, t)
        for i, (g, e, p, q, t) in enumerate(
            [
                (g, e, p, q, t)
                for g in ["male", "female"]
                for e in ["white-british", "south-asian"]
                for p in ["high", "low"]
                for q in ["high", "low"]
                for t in ["template_a", "template_b"]
            ]
        )
    ]


class MockModel:
    """Returns scores; optional responder override for failure injection."""

    def __init__(self, responder=None):
        self.responder = responder
        self.prompts: list[str] = []

    def predict(self, prompt: str, temperature: float = 1.0, **kwargs) -> str:
        self.prompts.append(prompt)
        if "Say 'ready'" in prompt:
            return "ready"
        if self.responder is not None:
            return self.responder(prompt)
        return default_responder(prompt)

    @property
    def attempts(self) -> list[str]:
        """Non-warmup predict calls."""
        return [p for p in self.prompts if "Say 'ready'" not in p]


@pytest.fixture
def factory():
    def _make(responder=None):
        model = MockModel(responder)
        return lambda **kwargs: model, model

    return _make


class TestParseScore:
    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("85/100", 85),
            ("The score is 92/100.", 92),
            ("Score: 0/100", 0),
            ("100/100", 100),
            ("101/100", None),  # out of range
            ("1000/100", None),  # no word-boundary match
            ("no score here", None),
            ("abc/100", None),
        ],
    )
    def test_parse_score(self, output, expected):
        assert parse_score(output) == expected


class TestStratifiedSample:
    def test_none_returns_all(self, small_cvs):
        assert stratified_sample(small_cvs, None) is small_cvs

    def test_max_ge_len_returns_all(self, small_cvs):
        assert stratified_sample(small_cvs, 1000) is small_cvs

    def test_zero_returns_empty(self, small_cvs):
        assert stratified_sample(small_cvs, 0) == []

    def test_length(self, small_cvs):
        assert len(stratified_sample(small_cvs, 8)) == 8

    def test_deterministic(self, small_cvs):
        first = stratified_sample(small_cvs, 10)
        second = stratified_sample(small_cvs, 10)
        assert [cv["cv"] for cv in first] == [cv["cv"] for cv in second]

    def test_balances_factors(self, small_cvs):
        sample = stratified_sample(small_cvs, 8)
        for factor in ("name_gender", "name_ethnicity", "university_prestige"):
            assert len({cv["metadata"][factor] for cv in sample}) == 2

    def test_preserves_original_order(self, small_cvs):
        sample = stratified_sample(small_cvs, 8)
        positions = [small_cvs.index(cv) for cv in sample]
        assert positions == sorted(positions)

    def test_prefix_slice_was_biased(self, small_cvs):
        """The bug this replaces: a prefix slice shows only one gender."""
        prefix = small_cvs[:4]
        assert {cv["metadata"]["name_gender"] for cv in prefix} == {"male"}
        sample = stratified_sample(small_cvs, 4)
        assert {cv["metadata"]["name_gender"] for cv in sample} == {"male", "female"}


class TestAttempt:
    def test_success_marks_seen(self, small_cvs):
        model = MockModel()
        seen = set()
        status, record = process_cv_run(model, small_cvs[0], 0, build_base_prompt(JOB_DESC), seen)
        assert status == STATUS_OK
        assert record is not None and record["score"] == 70
        assert record["response"] == "70/100"
        assert len(seen) == 1

    def test_parse_failure_not_marked_seen(self, small_cvs):
        model = MockModel(responder=lambda p: "garbage")
        seen = set()
        status, record = process_cv_run(model, small_cvs[0], 0, build_base_prompt(JOB_DESC), seen)
        assert status == STATUS_PARSE_ERROR
        assert record is None
        assert seen == set()  # retried on next invocation

    def test_api_error_not_marked_seen(self, small_cvs):
        def boom(prompt):
            raise ConnectionError("down")

        model = MockModel(responder=boom)
        seen = set()
        status, record = process_cv_run(model, small_cvs[0], 0, build_base_prompt(JOB_DESC), seen)
        assert status == STATUS_API_ERROR
        assert record is None
        assert seen == set()

    def test_skips_already_seen(self, small_cvs):
        base = build_base_prompt(JOB_DESC)
        key = sha256_hash(cv_prompt(base, small_cvs[0]))
        model = MockModel()
        status, record = process_cv_run(model, small_cvs[0], 0, base, {(key, 0)})
        assert status == STATUS_SKIPPED
        assert record is None
        assert model.attempts == []


class TestRunBenchmark:
    def _run(self, output_dir, cv_data, factory, **kwargs):
        model_factory, model = factory()
        df = run_cv_screening(
            model_name="mock-model",
            output_dir=str(output_dir),
            cv_data=cv_data,
            job_desc=JOB_DESC,
            model_factory=model_factory,
            **kwargs,
        )
        return df, model

    def test_full_run_writes_all_artefacts(self, tmp_path, small_cvs, factory):
        df, model = self._run(tmp_path, small_cvs, factory, n_runs=2, concurrency=1)

        assert len(df) == 64
        assert len(model.attempts) == 64
        assert (tmp_path / "records.csv").exists()
        assert (tmp_path / "analysis_summary.txt").exists()
        assert not (tmp_path / "records_checkpoint.jsonl").exists()
        assert list((tmp_path / "plots").glob("score_distribution_by_*.png"))

        raw = (tmp_path / "records.csv").read_text()
        assert "response" in raw.splitlines()[0]

    def test_cv_screening_json_schema(self, tmp_path, small_cvs, factory):
        df, _ = self._run(tmp_path, small_cvs, factory, n_runs=2, concurrency=1)

        payload = json.loads((tmp_path / "cv-screening.json").read_text())
        json.dumps(payload, allow_nan=False)  # strictly JSON-safe (no NaN/inf)

        assert payload["benchmark"] == "cv-screening"
        assert payload["n_examples"] == 64
        assert payload["mean_score"] == pytest.approx(df["score"].mean())

        attrition = payload["attrition"]
        assert attrition["n_planned"] == 64
        assert attrition["n_scored"] == 64
        assert attrition["n_outstanding"] == 0
        assert attrition["n_parse_failures_this_invocation"] == 0
        assert attrition["n_api_errors_this_invocation"] == 0
        assert attrition["n_records_total"] == 64

        provenance = payload["provenance"]
        assert provenance["model"] == "mock-model"
        assert provenance["package_version"] is not None
        assert provenance["n_runs"] == 2
        assert provenance["temperature"] == 1.0
        assert len(provenance["prompt_sha256"]) == 64
        assert provenance["timestamp"]

        gender = payload["groups"]["name_gender"]
        assert {row["name_gender"] for row in gender["summary"]} == {"male", "female"}
        assert all(70 <= row["mean"] <= 101 for row in gender["summary"])
        assert len(gender["pairwise"]) == 1
        assert "name_gender" in payload["variance_breakdown"]
        assert payload["variance_breakdown"]["name_gender"]["proportion"] > 0

    def test_artefacts_regenerated_on_fully_resumed_run(self, tmp_path, small_cvs, factory):
        self._run(tmp_path, small_cvs, factory, n_runs=1, concurrency=1)
        (tmp_path / "cv-screening.json").unlink()
        (tmp_path / "analysis_summary.txt").unlink()

        df, model = self._run(tmp_path, small_cvs, factory, n_runs=1, concurrency=1)

        assert len(df) == 32
        assert model.attempts == []  # nothing new to score
        assert (tmp_path / "cv-screening.json").exists()
        assert (tmp_path / "analysis_summary.txt").exists()

    def test_parse_failures_counted_in_attrition(self, tmp_path, small_cvs, factory):
        df, _ = self._run(
            tmp_path,
            small_cvs,
            lambda: factory(lambda p: "no score" if "ID=3 gender" in p else "75/100"),
            n_runs=1,
            concurrency=1,
        )
        assert len(df) == 31
        payload = json.loads((tmp_path / "cv-screening.json").read_text())
        assert payload["attrition"]["n_planned"] == 32
        assert payload["attrition"]["n_scored"] == 31
        assert payload["attrition"]["n_outstanding"] == 1
        assert payload["attrition"]["n_parse_failures_this_invocation"] == 1
        assert payload["attrition"]["n_records_total"] == 31

    def test_all_failures_yields_empty_run(self, tmp_path, small_cvs, factory):
        df, _ = self._run(
            tmp_path,
            small_cvs,
            lambda: factory(lambda p: "garbage"),
            n_runs=1,
            concurrency=1,
        )
        assert df.empty
        assert not (tmp_path / "records.csv").exists()

        payload = json.loads((tmp_path / "cv-screening.json").read_text())
        assert payload["n_examples"] == 0
        assert payload["mean_score"] is None
        assert payload["groups"] == {}
        assert payload["attrition"] == {
            "n_planned": 32,
            "n_scored": 0,
            "n_outstanding": 32,
            "n_parse_failures_this_invocation": 32,
            "n_api_errors_this_invocation": 0,
            "n_records_total": 0,
        }
        assert payload["provenance"]["prompt_sha256"]


class FakePool:
    """PoolClientProtocol double: routes prompts through an optional responder."""

    batch_size = 8
    num_ctx = 512
    keep_alive = 9.0

    def __init__(self, responder=None):
        self.responder = responder
        self.job_ids: list[str] = []

    def predict_batch(self, jobs):
        results = {}
        for job in jobs:
            self.job_ids.append(job["id"])
            if self.responder is None:
                results[job["id"]] = {
                    "response": default_responder(job["prompt"]),
                    "error": None,
                }
            else:
                results[job["id"]] = self.responder(job)
        return results

    def close(self) -> None:
        return None


class TestRunCvScreeningPool:
    """run_cv_screening dispatches to the Node.js pool with sequential-equivalent semantics."""

    def _run(self, output_dir, cv_data, pool, **kwargs):
        return run_cv_screening(
            model_name="mock-model",
            output_dir=str(output_dir),
            cv_data=cv_data,
            job_desc=JOB_DESC,
            pool_client=pool,
            **kwargs,
        )

    def test_pool_run_scores_every_attempt(self, tmp_path, small_cvs):
        pool = FakePool()
        df = self._run(tmp_path, small_cvs, pool, n_runs=2)

        assert len(df) == 64
        assert len(pool.job_ids) == 64
        assert (tmp_path / "records.csv").exists()
        assert not (tmp_path / "records_checkpoint.jsonl").exists()

        payload = json.loads((tmp_path / "cv-screening.json").read_text())
        assert payload["attrition"]["n_planned"] == 64
        assert payload["attrition"]["n_scored"] == 64
        assert payload["provenance"]["num_ctx"] == 512
        assert payload["provenance"]["keep_alive"] == 9.0

    def test_pool_keeps_raw_response(self, tmp_path, small_cvs):
        df = self._run(tmp_path, small_cvs, FakePool(), n_runs=1)
        assert "response" in df.columns
        assert df["response"].str.contains("/100").all()

    def test_pool_resume_skips_completed(self, tmp_path, small_cvs):
        self._run(tmp_path, small_cvs, FakePool(), n_runs=2)
        resumed = FakePool()
        df = self._run(tmp_path, small_cvs, resumed, n_runs=2)
        assert resumed.job_ids == []
        assert len(df) == 64

    def test_pool_errors_not_marked_seen(self, tmp_path, small_cvs):
        pool = FakePool(lambda job: {"response": None, "error": "boom"})
        df = self._run(tmp_path, small_cvs, pool, n_runs=1)
        assert df.empty

        payload = json.loads((tmp_path / "cv-screening.json").read_text())
        assert payload["attrition"]["n_api_errors_this_invocation"] == 32
        assert payload["attrition"]["n_outstanding"] == 32

    def test_pool_parse_failures_retry_next_invocation(self, tmp_path, small_cvs):
        df = self._run(
            tmp_path, small_cvs, FakePool(lambda job: {"response": "nope", "error": None}), n_runs=1
        )
        assert df.empty

        retry = FakePool()
        df = self._run(tmp_path, small_cvs, retry, n_runs=1)
        assert len(retry.job_ids) == 32  # failures were not marked seen
        assert len(df) == 32


class SystemOneStub:
    """Predictor double for the systemone path: records prompts, fixed score line."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def predict(self, prompt: str, temperature: float = 1.0) -> str:
        self.prompts.append(prompt)
        return '{"x":1}\n47/100'


class TestSystemOnePath:
    """api="systemone" runs the decision instrument sequentially, no chat warmup."""

    def _run(self, output_dir, cv_data, stub, **kwargs):
        return run_cv_screening(
            model_name="laya-english",
            output_dir=str(output_dir),
            cv_data=cv_data,
            job_desc=JOB_DESC,
            api="systemone",
            model_factory=lambda **factory_kwargs: stub,
            **kwargs,
        )

    def test_scores_with_decision_state_prompts(self, tmp_path, small_cvs):
        stub = SystemOneStub()
        df = self._run(tmp_path, small_cvs, stub, n_runs=1)

        assert len(df) == 32
        assert (df["score"] == 47).all()
        assert df["response"].str.endswith("47/100").all()
        assert stub.prompts == [decision_state(cv) for cv in small_cvs]

        payload = json.loads((tmp_path / "cv-screening.json").read_text())
        provenance = payload["provenance"]
        assert provenance["api"] == "systemone"
        assert provenance["temperature"] is None
        assert provenance["prompt_sha256"] == sha256_hash(decision_base_frame())

    def test_pool_client_rejected(self, tmp_path, small_cvs):
        with pytest.raises(ValueError, match="chat-only"):
            self._run(tmp_path, small_cvs, SystemOneStub(), pool_client=FakePool())

    def test_invalid_api_rejected(self, tmp_path, small_cvs):
        with pytest.raises(ValueError, match="unsupported api"):
            run_cv_screening(
                model_name="mock-model",
                output_dir=str(tmp_path),
                cv_data=small_cvs,
                job_desc=JOB_DESC,
                api="openai",
            )
