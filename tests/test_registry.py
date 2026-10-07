import pytest

from slm_bias_testing.registry import MODELS, get_model


class TestGetModel:
    def test_get_model_valid(self):
        model = get_model("smollm-135m")
        assert model["ollama_tag"] == "smollm:135m"
        assert model["params"] == 135_000_000
        assert model["family"] == "huggingface"

    def test_get_model_invalid(self):
        with pytest.raises(KeyError):
            get_model("nonexistent-model")


class TestRegistryImmutability:
    def test_get_model_returns_copy(self):
        model = get_model("smollm-135m")
        model["ollama_tag"] = "mutated:tag"
        model["params"] = -1
        fresh = get_model("smollm-135m")
        assert fresh["ollama_tag"] == "smollm:135m"
        assert fresh["params"] == 135_000_000
        assert MODELS["smollm-135m"]["params"] == 135_000_000

    def test_modes_is_read_only(self):
        with pytest.raises(TypeError):
            MODELS["new-model"] = get_model("smollm-135m")  # ty: ignore[invalid-assignment]

    def test_registry_entries_are_read_only(self):
        with pytest.raises(TypeError):
            MODELS["smollm-135m"]["params"] = -1


VALID_ARCHS = {"decoder-only", "hybrid-conv-attn", "encoder-only"}


class TestRegistryCompleteness:
    def test_all_models_have_required_fields(self):
        required = {"ollama_tag", "params", "release_date", "family", "architecture", "api"}
        for name, config in MODELS.items():
            assert required.issubset(config.keys()), (
                f"Model {name} missing fields: {required - set(config.keys())}"
            )

    def test_all_params_are_positive(self):
        for name, config in MODELS.items():
            assert config["params"] > 0, f"Model {name} has non-positive params"

    def test_all_models_known_architecture(self):
        for name, config in MODELS.items():
            assert config["architecture"] in VALID_ARCHS, (
                f"Model {name} has unexpected architecture: {config['architecture']}"
            )


LAYA_MODELS = ("laya-english", "laya-multilingual", "laya-typed-decisions")


class TestApiField:
    def test_every_entry_declares_api(self):
        for name, config in MODELS.items():
            assert config["api"] in ("chat", "systemone"), f"Model {name} has bad api"

    def test_non_laya_entries_are_chat(self):
        for name, config in MODELS.items():
            if name not in LAYA_MODELS:
                assert config["api"] == "chat", f"Model {name} should be chat"

    def test_laya_entries_exist_and_are_systemone(self):
        for name in LAYA_MODELS:
            assert name in MODELS
            assert MODELS[name]["api"] == "systemone"

    def test_laya_tags_are_unique(self):
        tags = [MODELS[name]["ollama_tag"] for name in LAYA_MODELS]
        assert len(set(tags)) == len(LAYA_MODELS)

    def test_laya_metadata_facts(self):
        expected = {
            "laya-english": ("laya:421m-english-mlx-fp16", 421_000_000),
            "laya-multilingual": ("laya:322m-multilingual-mlx-fp16", 322_000_000),
            "laya-typed-decisions": ("laya:421m-typed-decisions-mlx-fp16", 421_000_000),
        }
        for name, (tag, params) in expected.items():
            config = MODELS[name]
            assert config["ollama_tag"] == tag
            assert config["params"] == params
            assert config["release_date"] == "2026-09"
            assert config["family"] == "convai"
            assert config["architecture"] == "encoder-only"
