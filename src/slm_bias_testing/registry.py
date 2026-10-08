"""Model registry — single source of truth for model metadata.

Every model list in the codebase (runner, temporal analysis, tooling)
derives from :data:`MODELS`; nothing keeps its own copy.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Mapping


class ModelMeta(TypedDict):
    ollama_tag: str
    params: int
    release_date: str  # "YYYY-MM"
    family: str
    architecture: str
    api: Literal["chat", "systemone"]


_MODELS: dict[str, ModelMeta] = {
    "smollm-135m": {
        "ollama_tag": "smollm:135m",
        "params": 135_000_000,
        "release_date": "2024-07",
        "family": "huggingface",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "smollm-360m": {
        "ollama_tag": "smollm:360m",
        "params": 360_000_000,
        "release_date": "2024-07",
        "family": "huggingface",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "smollm2-135m": {
        "ollama_tag": "smollm2:135m",
        "params": 135_000_000,
        "release_date": "2024-11",
        "family": "huggingface",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "smollm2-360m": {
        "ollama_tag": "smollm2:360m",
        "params": 360_000_000,
        "release_date": "2024-11",
        "family": "huggingface",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "qwen25-05b": {
        "ollama_tag": "qwen2.5:0.5b",
        "params": 500_000_000,
        "release_date": "2024-09",
        "family": "alibaba",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "qwen25-15b": {
        "ollama_tag": "qwen2.5:1.5b",
        "params": 1_500_000_000,
        "release_date": "2024-09",
        "family": "alibaba",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "qwen35-08b": {
        "ollama_tag": "qwen3.5:0.8b",
        "params": 800_000_000,
        "release_date": "2025-05",
        "family": "alibaba",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "qwen3-06b": {
        "ollama_tag": "qwen3:0.6b",
        "params": 600_000_000,
        "release_date": "2025-04",
        "family": "alibaba",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "gemma3-270m": {
        "ollama_tag": "gemma3:270m",
        "params": 270_000_000,
        "release_date": "2025-03",
        "family": "google",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "granite4-350m": {
        "ollama_tag": "granite4:350m",
        "params": 350_000_000,
        "release_date": "2025-10",
        "family": "ibm",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "lfm2-350m": {
        "ollama_tag": "sam860/lfm2:350m",
        "params": 350_000_000,
        "release_date": "2025-07",
        "family": "liquid",
        "architecture": "hybrid-conv-attn",
        "api": "chat",
    },
    "lfm2-700m": {
        "ollama_tag": "sam860/lfm2:700m",
        "params": 700_000_000,
        "release_date": "2025-07",
        "family": "liquid",
        "architecture": "hybrid-conv-attn",
        "api": "chat",
    },
    "llama32-1b": {
        "ollama_tag": "llama3.2:1b",
        "params": 1_000_000_000,
        "release_date": "2024-09",
        "family": "meta",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "tinyllama": {
        "ollama_tag": "tinyllama",
        "params": 1_100_000_000,
        "release_date": "2023-12",
        "family": "meta",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "stablelm2-16b": {
        "ollama_tag": "stablelm2:1.6b",
        "params": 1_600_000_000,
        "release_date": "2024-01",
        "family": "stability",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "gemma3-1b": {
        "ollama_tag": "gemma3:1b-it-qat",
        "params": 1_000_000_000,
        "release_date": "2025-03",
        "family": "google",
        "architecture": "decoder-only",
        "api": "chat",
    },
    "laya-english": {
        "ollama_tag": "laya:421m-english-mlx-fp16",
        "params": 421_000_000,
        "release_date": "2026-09",
        "family": "convai",
        "architecture": "encoder-only",
        "api": "systemone",
    },
    "laya-multilingual": {
        "ollama_tag": "laya:322m-multilingual-mlx-fp16",
        "params": 322_000_000,
        "release_date": "2026-09",
        "family": "convai",
        "architecture": "encoder-only",
        "api": "systemone",
    },
    "laya-typed-decisions": {
        "ollama_tag": "laya:421m-typed-decisions-mlx-fp16",
        "params": 421_000_000,
        "release_date": "2026-09",
        "family": "convai",
        "architecture": "encoder-only",
        "api": "systemone",
    },
    "nimble": {
        "ollama_tag": "nimble",
        "params": 9_000_000_000,
        "release_date": "2026-09",
        "family": "bespoke",
        "architecture": "decoder-only",
        "api": "systemone",
    },
}


#: Read-only view over the registry: neither the mapping nor any entry can
#: be mutated. ``get_model()`` still hands out mutable copies.
MODELS: Mapping[str, ModelMeta] = MappingProxyType(
    {name: cast("ModelMeta", MappingProxyType(dict(meta))) for name, meta in _MODELS.items()}
)


def get_model(name: str) -> ModelMeta:
    """Get a model's metadata as a copy. Raises KeyError if not found.

    The copy is intentional — mutating a returned config must never affect
    other callers.
    """
    return MODELS[name].copy()
