import tempfile
from unittest.mock import patch

import pytest

from slm_bias_testing.benchmarks import BaseBenchmark
from slm_bias_testing.benchmarks.demographic_bias import DemographicBiasBenchmark
from slm_bias_testing.benchmarks.stereoset import StereoSetBenchmark
from slm_bias_testing.benchmarks.winobias import WinoBiasBenchmark


class TestBaseBenchmark:
    def test_interface(self):
        class ConcreteBench(BaseBenchmark):
            name = "test"

            def load_dataset(self):
                return []

            def evaluate(self, model=None, max_samples=None, output_dir=None, pool_client=None):
                return {"result": 42}

        bm = ConcreteBench()
        assert bm.load_dataset() == []
        assert bm.evaluate(None) == {"result": 42}

    def test_save_and_load_results(self):
        class ConcreteBench(BaseBenchmark):
            name = "test"

            def load_dataset(self):
                return []

            def evaluate(self, model=None, max_samples=None, output_dir=None, pool_client=None):
                return {"result": 42}

        bm = ConcreteBench()
        with tempfile.TemporaryDirectory() as tmpdir:
            bm.save_results({"result": 42}, tmpdir)
            loaded = bm.load_results(tmpdir)
            assert loaded == {"result": 42}

    def test_load_results_nonexistent(self):
        class ConcreteBench(BaseBenchmark):
            name = "test"

            def load_dataset(self):
                return []

            def evaluate(self, model=None, max_samples=None, output_dir=None, pool_client=None):
                return {"result": 42}

        bm = ConcreteBench()
        with tempfile.TemporaryDirectory() as tmpdir:
            assert bm.load_results(tmpdir) is None


class MockPoolClient:
    """Mock pool client that returns pre-configured responses."""

    def __init__(self, responses: dict[str, str] | None = None, batch_size: int = 40):
        self.responses = responses or {}
        self.batch_size = batch_size
        self.model_name = "test-model"
        self.pool_size = batch_size
        self.batch_timeout = 300

    def predict_batch(self, jobs: list[dict]) -> dict[str, dict]:
        results = {}
        for job in jobs:
            prompt = job["prompt"]
            response = "50"
            for key, val in self.responses.items():
                if key in prompt:
                    response = val
                    break
            results[job["id"]] = {"response": response, "error": None}
        return results

    def close(self) -> None:
        pass


def _make_stereoset_item(id, bias_type, target, context, stereotype, anti_stereotype):
    """Helper to create a StereoSet item in the real nested format."""
    return {
        "id": id,
        "bias_type": bias_type,
        "target": target,
        "context": context,
        "sentences": {
            "sentence": [stereotype, "unrelated sentence", anti_stereotype],
            "gold_label": [1, 0, 2],  # 1=stereotype, 0=unrelated, 2=anti-stereotype
        },
    }


class TestStereoSetBenchmark:
    def test_load_dataset(self):
        fake_data = [
            _make_stereoset_item(
                "test1",
                "gender",
                "doctor",
                "The doctor entered.",
                "He is skilled.",
                "She is skilled.",
            )
        ]
        with patch("slm_bias_testing.benchmarks.stereoset.datasets.load_dataset") as mock_load:
            mock_load.return_value = fake_data
            bm = StereoSetBenchmark()
            data = bm.load_dataset()
            assert len(data) == 1
            assert data[0]["bias_type"] == "gender"
            mock_load.assert_called_once_with("stereoset", "intrasentence", split="validation")

    def test_evaluate_chosen_stereotype(self):
        fake_data = [
            _make_stereoset_item(
                "test1",
                "gender",
                "doctor",
                "The doctor entered.",
                "He is skilled.",
                "She is skilled.",
            )
        ]
        pool = MockPoolClient({"He is skilled.": "90", "She is skilled.": "10"})
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(None, pool_client=pool)
            assert results["benchmark"] == "stereoset"
            assert results["n_examples"] == 1
            assert results["overall_stereotype_score"] == 100.0

    def test_evaluate_chosen_anti_stereotype(self):
        fake_data = [
            _make_stereoset_item(
                "test1",
                "gender",
                "doctor",
                "The doctor entered.",
                "He is skilled.",
                "She is skilled.",
            )
        ]
        pool = MockPoolClient({"He is skilled.": "10", "She is skilled.": "90"})
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(None, pool_client=pool)
            assert results["overall_stereotype_score"] == 0.0

    def test_per_category(self):
        fake_data = [
            _make_stereoset_item(
                "test1",
                "gender",
                "doctor",
                "The doctor entered.",
                "He is skilled.",
                "She is skilled.",
            ),
            _make_stereoset_item(
                "test2",
                "race",
                "person",
                "The person walked in.",
                "He is loud.",
                "She is quiet.",
            ),
        ]
        pool = MockPoolClient(
            {
                "He is skilled.": "90",
                "She is skilled.": "10",
                "He is loud.": "80",
                "She is quiet.": "20",
            }
        )
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(None, pool_client=pool)
            assert results["per_category"]["gender"] == 100.0
            assert results["per_category"]["race"] == 100.0

    def test_max_samples_limits_evaluation(self):
        fake_data = [
            _make_stereoset_item(
                "test1",
                "gender",
                "doctor",
                "The doctor entered.",
                "He is skilled.",
                "She is skilled.",
            ),
            _make_stereoset_item(
                "test2",
                "race",
                "person",
                "The person walked in.",
                "He is loud.",
                "She is quiet.",
            ),
            _make_stereoset_item(
                "test3",
                "age",
                "worker",
                "The worker arrived.",
                "He is experienced.",
                "She is experienced.",
            ),
        ]
        pool = MockPoolClient({"He is skilled.": "90", "She is skilled.": "10"})
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(None, max_samples=1, pool_client=pool)
            assert results["n_examples"] == 1

    def test_missing_gold_label_skipped(self):
        fake_data = [
            {
                "id": "bad1",
                "bias_type": "gender",
                "target": "doctor",
                "context": "The doctor entered.",
                "sentences": {"sentence": ["Only unrelated here"], "gold_label": [0]},
            },
            _make_stereoset_item(
                "good1",
                "gender",
                "doctor",
                "The doctor entered.",
                "He is skilled.",
                "She is skilled.",
            ),
        ]
        pool = MockPoolClient({"He is skilled.": "90", "She is skilled.": "10"})
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(None, pool_client=pool)
            assert results["n_examples"] == 1


class TestStereoSetPool:
    """Tests for StereoSet pool batch mode."""

    def test_evaluate_pool_chosen_stereotype(self):
        fake_data = [
            _make_stereoset_item(
                "test1",
                "gender",
                "doctor",
                "The doctor entered.",
                "He is skilled.",
                "She is skilled.",
            )
        ]
        pool = MockPoolClient({"He is skilled.": "90", "She is skilled.": "10"})
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(None, pool_client=pool)
            assert results["benchmark"] == "stereoset"
            assert results["n_examples"] == 1
            assert results["overall_stereotype_score"] == 100.0

    def test_evaluate_pool_per_category(self):
        fake_data = [
            _make_stereoset_item(
                "test1",
                "gender",
                "doctor",
                "The doctor entered.",
                "He is skilled.",
                "She is skilled.",
            ),
            _make_stereoset_item(
                "test2",
                "race",
                "person",
                "The person walked in.",
                "He is loud.",
                "She is quiet.",
            ),
        ]
        pool = MockPoolClient(
            {
                "He is skilled.": "90",
                "She is skilled.": "10",
                "He is loud.": "80",
                "She is quiet.": "20",
            }
        )
        with patch.object(StereoSetBenchmark, "load_dataset", return_value=fake_data):
            bm = StereoSetBenchmark()
            results = bm.evaluate(None, pool_client=pool)
            assert results["per_category"]["gender"] == 100.0
            assert results["per_category"]["race"] == 100.0


class TestWinoBiasBenchmark:
    @pytest.mark.integration
    def test_load_dataset(self):
        bm = WinoBiasBenchmark()
        data = bm.load_dataset()
        assert len(data) == 1584
        configs = set(d["config"] for d in data)
        assert configs == {"type1_pro", "type1_anti", "type2_pro", "type2_anti"}

    @pytest.mark.integration
    def test_occupations_set(self):
        bm = WinoBiasBenchmark()
        bm.load_dataset()
        occ = bm._get_occupations()
        assert "developer" in occ
        assert "nurse" in occ
        assert len(occ) == 40

    @pytest.mark.integration
    def test_entity_extraction(self):
        bm = WinoBiasBenchmark()
        data = bm.load_dataset()
        item = next(d for d in data if d["config"] == "type1_pro")
        tokens = item["tokens"]
        coref = [int(x) for x in item["coreference_clusters"]]
        entities = bm._find_entities(tokens, coref[2])
        assert len(entities) == 2
        correct = bm._extract_entity_name(tokens, coref[0], coref[1])
        assert correct in entities

    def test_compute_metrics(self):
        bm = WinoBiasBenchmark()
        fake_results = [
            {"correct": True, "is_pro": True, "config": "type1_pro", "pronoun": "he"},
            {"correct": False, "is_pro": False, "config": "type1_anti", "pronoun": "she"},
            {"correct": True, "is_pro": True, "config": "type1_pro", "pronoun": "he"},
            {"correct": True, "is_pro": False, "config": "type1_anti", "pronoun": "she"},
        ]
        metrics = bm._compute_metrics(fake_results)
        assert metrics["overall_accuracy"] == 75.0
        assert metrics["pro_accuracy"] == 100.0
        assert metrics["anti_accuracy"] == 50.0
        assert metrics["bias_score"] == 50.0
        assert metrics["n_examples"] == 4


class TestDemographicBiasBenchmark:
    def test_load_dataset(self):
        bm = DemographicBiasBenchmark()
        data = bm.load_dataset()
        assert len(data) > 0
        assert "prompt" in data[0]
        assert "group" in data[0]
        assert "term" in data[0]

    def test_groups_covered(self):
        bm = DemographicBiasBenchmark()
        data = bm.load_dataset()
        groups = set(d["group"] for d in data)
        assert "gender_male" in groups
        assert "gender_female" in groups
        assert "race_white" in groups
        assert "race_black" in groups

    def test_evaluate(self):
        pool = MockPoolClient({"The man walked": "Hello", "The woman walked": "World"})
        with patch.object(
            DemographicBiasBenchmark,
            "load_dataset",
            return_value=[
                {
                    "prompt": "The man walked into the room and",
                    "group": "gender_male",
                    "term": "man",
                },
                {
                    "prompt": "The woman walked into the room and",
                    "group": "gender_female",
                    "term": "woman",
                },
            ],
        ):
            bm = DemographicBiasBenchmark()
            results = bm.evaluate(None, pool_client=pool)
            assert results["benchmark"] == "demographic-bias"
            assert results["n_examples"] == 2
            assert "gender_male" in results["per_group"]
            assert "gender_female" in results["per_group"]


class TestDemographicBiasPool:
    def test_evaluate_pool(self):
        pool = MockPoolClient({"The man walked": "Hello", "The woman walked": "World"})
        with patch.object(
            DemographicBiasBenchmark,
            "load_dataset",
            return_value=[
                {
                    "prompt": "The man walked into the room and",
                    "group": "gender_male",
                    "term": "man",
                },
                {
                    "prompt": "The woman walked into the room and",
                    "group": "gender_female",
                    "term": "woman",
                },
            ],
        ):
            bm = DemographicBiasBenchmark()
            results = bm.evaluate(None, pool_client=pool)
            assert results["benchmark"] == "demographic-bias"
            assert results["n_examples"] == 2
            assert "gender_male" in results["per_group"]
            assert "gender_female" in results["per_group"]


def _make_winobias_item(idx, config, tokens, coref_clusters, is_pro=True):
    return {
        "tokens": tokens,
        "coreference_clusters": coref_clusters,
        "config": config,
    }


class TestWinoBiasPool:
    def test_evaluate_pool_basic(self):
        fake_data = [
            _make_winobias_item(
                0,
                "type1_pro",
                ["The", "developer", "told", "the", "nurse", "that", "she", "was", "late"],
                ["3", "4", "6"],
            ),
        ]
        pool = MockPoolClient({"she": "nurse"})
        with (
            patch.object(WinoBiasBenchmark, "load_dataset", return_value=fake_data),
            patch.object(
                WinoBiasBenchmark, "_get_occupations", return_value={"developer", "nurse"}
            ),
        ):
            bm = WinoBiasBenchmark()
            results = bm.evaluate(None, pool_client=pool)
            assert results["benchmark"] == "winobias"
            assert results["n_examples"] == 1
            assert results["overall_accuracy"] == 100.0

    def test_evaluate_pool_incorrect_answer(self):
        fake_data = [
            _make_winobias_item(
                0,
                "type1_pro",
                ["The", "developer", "told", "the", "nurse", "that", "she", "was", "late"],
                ["3", "4", "6"],
            ),
        ]
        pool = MockPoolClient({"she": "developer"})
        with (
            patch.object(WinoBiasBenchmark, "load_dataset", return_value=fake_data),
            patch.object(
                WinoBiasBenchmark, "_get_occupations", return_value={"developer", "nurse"}
            ),
        ):
            bm = WinoBiasBenchmark()
            results = bm.evaluate(None, pool_client=pool)
            assert results["overall_accuracy"] == 0.0

    def test_evaluate_pool_empty_response(self):
        fake_data = [
            _make_winobias_item(
                0,
                "type1_pro",
                ["The", "developer", "told", "the", "nurse", "that", "she", "was", "late"],
                ["3", "4", "6"],
            ),
        ]
        pool = MockPoolClient({})
        with (
            patch.object(WinoBiasBenchmark, "load_dataset", return_value=fake_data),
            patch.object(
                WinoBiasBenchmark, "_get_occupations", return_value={"developer", "nurse"}
            ),
        ):
            bm = WinoBiasBenchmark()
            results = bm.evaluate(None, pool_client=pool)
            assert results["n_examples"] == 1

    def test_evaluate_pool_checkpoint_resume(self):
        import json
        import os
        import tempfile

        fake_data = [
            _make_winobias_item(
                0,
                "type1_pro",
                ["The", "developer", "told", "the", "nurse", "that", "she", "was", "late"],
                ["3", "4", "6"],
            ),
        ]
        pool = MockPoolClient({"she": "nurse"})
        with (
            patch.object(WinoBiasBenchmark, "load_dataset", return_value=fake_data),
            patch.object(
                WinoBiasBenchmark, "_get_occupations", return_value={"developer", "nurse"}
            ),
        ):
            bm = WinoBiasBenchmark()
            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_path = os.path.join(tmpdir, f"{bm.name}_checkpoint.jsonl")
                with open(ckpt_path, "w") as f:
                    f.write(
                        json.dumps(
                            {
                                "item_idx": 0,
                                "sentence": "The developer told the nurse that she was late",
                                "config": "type1_pro",
                                "pronoun": "she",
                                "entity1": "the developer",
                                "entity2": "the nurse",
                                "correct_antecedent": "the nurse",
                                "model_answer": "nurse",
                                "correct": True,
                                "is_pro": True,
                            }
                        )
                        + "\n"
                    )
                results = bm.evaluate(None, pool_client=pool, output_dir=tmpdir)
                assert results["n_examples"] == 1
                assert results["overall_accuracy"] == 100.0


class TestPoolClientRequired:
    """Verify that benchmarks raise ValueError when pool_client is None."""

    def test_stereoset_requires_pool_client(self):
        bm = StereoSetBenchmark()
        with pytest.raises(ValueError, match="pool_client"):
            bm.evaluate(None)

    def test_winobias_requires_pool_client(self):
        bm = WinoBiasBenchmark()
        with pytest.raises(ValueError, match="pool_client"):
            bm.evaluate(None)

    def test_demographic_bias_requires_pool_client(self):
        bm = DemographicBiasBenchmark()
        with pytest.raises(ValueError, match="pool_client"):
            bm.evaluate(None)
