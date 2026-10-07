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
            MODELS["new-model"] = get_model("smollm-135m")  # type: ignore[index]

    def test_registry_entries_are_read_only(self):
        with pytest.raises(TypeError):
            MODELS["smollm-135m"]["params"] = -1  # type: ignore[index]


VALID_ARCHS = {"decoder-only", "hybrid-conv-attn"}


class TestRegistryCompleteness:
    def test_all_models_have_required_fields(self):
        required = {"ollama_tag", "params", "release_date", "family", "architecture"}
        for name, config in MODELS.items():
            assert required.issubset(config.keys()), (
                f"Model {name} missing fields: {required - set(config.keys())}"
            )

    def test_all_params_are_positive(self):
        for name, config in MODELS.items():
            assert config["params"] > 0, f"Model {name} has non-positive params"

    def test_all_models_decoder_only(self):
        for name, config in MODELS.items():
            assert config["architecture"] in VALID_ARCHS, (
                f"Model {name} has unexpected architecture: {config['architecture']}"
            )
