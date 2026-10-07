#!/usr/bin/env python3
"""Latency / throughput spike for Ollama typed-decision models (/v1/systemone).

Measures, per model tag: cold load, warm latency (short control state and the
real CV-screening prompt), batched throughput, and the exact failure mode when
a state exceeds the model's context window.

Usage:
    uv run python scripts/benchmark_decision_model.py
    uv run python scripts/benchmark_decision_model.py \
        --models laya,laya:322m-multilingual-mlx-fp16 --n 50 --threads 4
    uv run python scripts/benchmark_decision_model.py --json /tmp/spike.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from slm_bias_testing.cv_screening import build_base_prompt, cv_prompt
from slm_bias_testing.data.cvs import cvs
from slm_bias_testing.data.job_description import job_description
from slm_bias_testing.decision_models import (
    CONTEXT_LIMITS,
    DEFAULT_HOST,
    build_cv_questions,
    build_short_questions,
    latency_stats,
    post_systemone,
    summarize_answers,
)

DEFAULT_MODELS = "laya,laya:322m-multilingual-mlx-fp16,laya:421m-typed-decisions-mlx-fp16"
SHORT_STATE = "Hello, how are you today?"


def real_cv_state() -> str:
    """The exact prompt CV screening sends to generative models."""
    return cv_prompt(build_base_prompt(job_description), cvs[0])


def fetch_tags(host: str) -> dict[str, dict[str, Any]]:
    """Map model name -> {digest, size} from GET /api/tags."""
    with urllib.request.urlopen(f"{host}/api/tags", timeout=30) as response:
        payload = json.loads(response.read())
    tags: dict[str, dict[str, Any]] = {}
    for model in payload.get("models", []):
        tags[str(model.get("name"))] = {
            "digest": str(model.get("digest", ""))[:12],
            "size": int(model.get("size", 0)),
        }
    return tags


def stop_model(model: str) -> None:
    """Unload a model so the next request measures a cold start."""
    subprocess.run(
        ["ollama", "stop", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def run_phase(
    model: str,
    state: str,
    questions: dict[str, Any],
    n: int,
    *,
    host: str,
    timeout: float,
) -> dict[str, Any]:
    """Run n sequential requests; return latencies, errors, tokens, sample."""
    latencies: list[float] = []
    errors: dict[str, int] = {}
    tokens: list[int] = []
    sample: dict[str, Any] | None = None
    for _ in range(n):
        latency, payload = post_systemone(model, state, questions, host=host, timeout=timeout)
        latencies.append(latency)
        if "error" in payload:
            message = str(payload["error"])
            errors[message] = errors.get(message, 0) + 1
        else:
            usage = payload.get("usage", {})
            if isinstance(usage.get("input_tokens"), int):
                tokens.append(int(usage["input_tokens"]))
            if sample is None:
                sample = summarize_answers(payload.get("answers", {}))
    return {
        "latency": latency_stats(latencies),
        "errors": errors,
        "input_tokens": latency_stats([float(t) for t in tokens]),
        "sample_answers": sample,
    }


def run_throughput(
    model: str,
    state: str,
    questions: dict[str, Any],
    n: int,
    threads: int,
    *,
    host: str,
    timeout: float,
) -> dict[str, Any]:
    """n requests across `threads` workers; report wall time and rps."""
    start = time.perf_counter()

    def one() -> tuple[float, dict[str, Any]]:
        return post_systemone(model, state, questions, host=host, timeout=timeout)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(lambda _: one(), range(n)))
    wall = time.perf_counter() - start
    errors = sum(1 for _, payload in results if "error" in payload)
    return {
        "requests": n,
        "threads": threads,
        "wall_s": round(wall, 3),
        "rps": round(n / wall, 2) if wall > 0 else 0.0,
        "errors": errors,
    }


def bench_model(
    model: str,
    *,
    host: str,
    n: int,
    threads: int,
    timeout: float,
    tags: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Full protocol for one model: cold, warm short, warm CV, throughput."""
    cv_state = real_cv_state()
    print(f"\n=== {model} ===", file=sys.stderr)
    stop_model(model)
    time.sleep(0.5)

    print("  cold...", file=sys.stderr)
    cold_latency, cold_payload = post_systemone(
        model, SHORT_STATE, build_short_questions(), host=host, timeout=timeout
    )
    cold_error = str(cold_payload.get("error")) if "error" in cold_payload else None

    print(f"  warm short x{n}...", file=sys.stderr)
    short = run_phase(
        model,
        SHORT_STATE,
        build_short_questions(),
        n,
        host=host,
        timeout=timeout,
    )

    print(f"  warm CV prompt x{n}...", file=sys.stderr)
    cv = run_phase(model, cv_state, build_cv_questions(), n, host=host, timeout=timeout)

    print(f"  throughput ({threads} threads x {n})...", file=sys.stderr)
    throughput = run_throughput(
        model,
        SHORT_STATE,
        build_short_questions(),
        n,
        threads,
        host=host,
        timeout=timeout,
    )
    stop_model(model)

    tag_info: dict[str, Any] = tags.get(model, {})
    if not tag_info and ":" not in model:
        tag_info = tags.get(f"{model}:latest", {})
    return {
        "model": model,
        "digest": tag_info.get("digest", "?"),
        "size_bytes": tag_info.get("size", 0),
        "context_limit": CONTEXT_LIMITS.get(model),
        "cold_s": round(cold_latency, 3),
        "cold_error": cold_error,
        "short": short,
        "cv": cv,
        "throughput": throughput,
    }


def print_table(results: list[dict[str, Any]]) -> None:
    hdr = (
        f"{'model':<40} {'ctx':>4} {'cold':>7} {'short p50':>9} {'short p95':>9} "
        f"{'cv p50':>7} {'cv p95':>7} {'in_tok':>6} {'rps':>7} {'cv_err':>6}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        short, cv = r["short"]["latency"], r["cv"]["latency"]
        cv_errs = sum(r["cv"]["errors"].values())
        cold = "ERR" if r["cold_error"] else f"{r['cold_s'] * 1000:.0f}ms"
        in_tok = r["cv"]["input_tokens"]["mean"]
        cv_p50 = f"{cv['p50'] * 1000:.0f}ms" if not cv_errs else "ERR"
        cv_p95 = f"{cv['p95'] * 1000:.0f}ms" if not cv_errs else "ERR"
        in_tok_s = f"{in_tok:.0f}" if in_tok else "-"
        ctx_s = str(r["context_limit"] or "?")
        print(
            f"{r['model']:<40} {ctx_s:>4} {cold:>7} "
            f"{short['p50'] * 1000:.0f}ms{'':>3} {short['p95'] * 1000:.0f}ms{'':>3} "
            f"{cv_p50:>7} {cv_p95:>7} {in_tok_s:>6} {r['throughput']['rps']:>7.1f} "
            f"{cv_errs:>6}"
        )
        if r["cold_error"]:
            print(f"    cold error: {r['cold_error'][:120]}")
        for message in list(r["cv"]["errors"])[:1]:
            print(f"    cv error: {message[:160]}")
        if r["cv"]["sample_answers"]:
            print(f"    cv answers: {r['cv']['sample_answers']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated tags")
    parser.add_argument("--n", type=int, default=50, help="requests per phase")
    parser.add_argument("--threads", type=int, default=4, help="throughput workers")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", dest="json_path", help="write full results JSON here")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tags = fetch_tags(args.host)
    started = datetime.now(UTC).isoformat(timespec="seconds")

    results = [
        bench_model(
            model,
            host=args.host,
            n=args.n,
            threads=args.threads,
            timeout=args.timeout,
            tags=tags,
        )
        for model in models
    ]

    report = {
        "meta": {
            "host": args.host,
            "n": args.n,
            "threads": args.threads,
            "started": started,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.platform(),
        },
        "results": results,
    }
    print_table(results)
    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
