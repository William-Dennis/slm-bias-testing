/* Model catalog — transcribed verbatim from
   src/slm_bias_testing/registry.py (single source of truth).
   Evaluation numbers built on top are synthetic demo data. */
import type { ModelMeta } from "./types";

export const MODELS: ModelMeta[] = [
  { id: "smollm-135m", ollamaTag: "smollm:135m", params: 135_000_000, releaseDate: "2024-07", family: "huggingface", architecture: "decoder-only", api: "chat" },
  { id: "smollm-360m", ollamaTag: "smollm:360m", params: 360_000_000, releaseDate: "2024-07", family: "huggingface", architecture: "decoder-only", api: "chat" },
  { id: "smollm2-135m", ollamaTag: "smollm2:135m", params: 135_000_000, releaseDate: "2024-11", family: "huggingface", architecture: "decoder-only", api: "chat" },
  { id: "smollm2-360m", ollamaTag: "smollm2:360m", params: 360_000_000, releaseDate: "2024-11", family: "huggingface", architecture: "decoder-only", api: "chat" },
  { id: "qwen25-05b", ollamaTag: "qwen2.5:0.5b", params: 500_000_000, releaseDate: "2024-09", family: "alibaba", architecture: "decoder-only", api: "chat" },
  { id: "qwen25-15b", ollamaTag: "qwen2.5:1.5b", params: 1_500_000_000, releaseDate: "2024-09", family: "alibaba", architecture: "decoder-only", api: "chat" },
  { id: "qwen35-08b", ollamaTag: "qwen3.5:0.8b", params: 800_000_000, releaseDate: "2025-05", family: "alibaba", architecture: "decoder-only", api: "chat" },
  { id: "qwen3-06b", ollamaTag: "qwen3:0.6b", params: 600_000_000, releaseDate: "2025-04", family: "alibaba", architecture: "decoder-only", api: "chat" },
  { id: "gemma3-270m", ollamaTag: "gemma3:270m", params: 270_000_000, releaseDate: "2025-03", family: "google", architecture: "decoder-only", api: "chat" },
  { id: "granite4-350m", ollamaTag: "granite4:350m", params: 350_000_000, releaseDate: "2025-10", family: "ibm", architecture: "decoder-only", api: "chat" },
  { id: "lfm2-350m", ollamaTag: "sam860/lfm2:350m", params: 350_000_000, releaseDate: "2025-07", family: "liquid", architecture: "hybrid-conv-attn", api: "chat" },
  { id: "lfm2-700m", ollamaTag: "sam860/lfm2:700m", params: 700_000_000, releaseDate: "2025-07", family: "liquid", architecture: "hybrid-conv-attn", api: "chat" },
  { id: "llama32-1b", ollamaTag: "llama3.2:1b", params: 1_000_000_000, releaseDate: "2024-09", family: "meta", architecture: "decoder-only", api: "chat" },
  { id: "tinyllama", ollamaTag: "tinyllama", params: 1_100_000_000, releaseDate: "2023-12", family: "meta", architecture: "decoder-only", api: "chat" },
  { id: "stablelm2-16b", ollamaTag: "stablelm2:1.6b", params: 1_600_000_000, releaseDate: "2024-01", family: "stability", architecture: "decoder-only", api: "chat" },
  { id: "gemma3-1b", ollamaTag: "gemma3:1b-it-qat", params: 1_000_000_000, releaseDate: "2025-03", family: "google", architecture: "decoder-only", api: "chat" },
  { id: "laya-english", ollamaTag: "laya:421m-english-mlx-fp16", params: 421_000_000, releaseDate: "2026-09", family: "convai", architecture: "encoder-only", api: "systemone" },
  { id: "laya-multilingual", ollamaTag: "laya:322m-multilingual-mlx-fp16", params: 322_000_000, releaseDate: "2026-09", family: "convai", architecture: "encoder-only", api: "systemone" },
  { id: "laya-typed-decisions", ollamaTag: "laya:421m-typed-decisions-mlx-fp16", params: 421_000_000, releaseDate: "2026-09", family: "convai", architecture: "encoder-only", api: "systemone" },
  { id: "nimble", ollamaTag: "nimble", params: 9_000_000_000, releaseDate: "2026-09", family: "bespoke", architecture: "decoder-only", api: "systemone" },
];

export const modelById = (id: string): ModelMeta =>
  MODELS.find((m) => m.id === id) ?? MODELS[0];

export function formatParams(p: number): string {
  if (p >= 1_000_000_000) return `${(p / 1_000_000_000).toFixed(1)}B`;
  return `${Math.round(p / 1_000_000)}M`;
}
