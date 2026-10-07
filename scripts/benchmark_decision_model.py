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
import os
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
from slm_bias_testing.decision_instrument import (
    BUDGET_TOKENS,
    decision_state,
    parse_state_tokens,
    score_from_answer,
    score_question,
)
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


def real_cv_state_cv(cv: dict[str, Any]) -> str:
    """The exact prompt for an arbitrary corpus CV."""
    return cv_prompt(build_base_prompt(job_description), cv)


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


def stop_model(model: str, host: str) -> None:
    """Unload a model so the next request measures a cold start.

    The CLI resolves its server from ``OLLAMA_HOST``, so the selected host
    must be forwarded — otherwise ``--host`` would stop a model on the
    wrong server and leave the cold measurement warm. Failures are
    non-fatal (the model may not be loaded).
    """
    subprocess.run(
        ["ollama", "stop", model],
        env={**os.environ, "OLLAMA_HOST": host},
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
    successes = n - errors
    return {
        "requests": n,
        "threads": threads,
        "wall_s": round(wall, 3),
        "rps_ok": round(successes / wall, 2) if wall > 0 else 0.0,
        "rps_attempted": round(n / wall, 2) if wall > 0 else 0.0,
        "errors": errors,
    }


def measure_question_overhead(model: str, *, host: str, timeout: float) -> int | None:
    """Question-only token overhead for this tag (state = 1 token "x").

    ``None`` when the probe fails - callers must not guess an overhead of 0,
    which would over-count every ``state_tokens`` value.
    """
    _, payload = post_systemone(model, "x", score_question(), host=host, timeout=timeout)
    used = payload.get("usage", {}).get("input_tokens") if "error" not in payload else None
    return max(0, int(used) - 1) if isinstance(used, int) else None


def score_sweep_model(
    model: str,
    cv_list: list[dict[str, Any]],
    *,
    host: str,
    timeout: float,
) -> dict[str, Any]:
    """Score every corpus CV with the v2 instrument on one model tag.

    Model stays resident for the whole sweep (one load, 600 decisions).
    """
    stop_model(model, host)
    time.sleep(0.5)
    overhead = measure_question_overhead(model, host=host, timeout=timeout)
    question = score_question()
    records: list[dict[str, Any]] = []
    n_ok = n_ctx_errors = n_request_errors = n_unscored = n_budget = 0
    started = time.perf_counter()
    for index, cv in enumerate(cv_list):
        meta = cv.get("metadata", {})
        state = decision_state(cv)
        latency, payload = post_systemone(model, state, question, host=host, timeout=timeout)
        record: dict[str, Any] = {
            "cv_index": index,
            "name": meta.get("name"),
            "name_gender": meta.get("name_gender"),
            "name_ethnicity": meta.get("name_ethnicity"),
            "university_prestige": meta.get("university_prestige"),
            "a_level_quality": meta.get("a_level_quality"),
            "template_name": meta.get("template_name"),
            "latency_s": round(latency, 4),
        }
        if "error" in payload:
            message = str(payload["error"])
            state_tokens = parse_state_tokens(message)
            record.update(
                {
                    "state_tokens": state_tokens,
                    "budget_ok": (
                        state_tokens <= BUDGET_TOKENS if state_tokens is not None else None
                    ),
                    "ctx_ok": False,
                    "score_discrete": None,
                    "score_continuous": None,
                    "probabilities": {},
                    "confidence": None,
                    "error": message,
                }
            )
            if state_tokens is not None:
                n_ctx_errors += 1
            else:
                n_request_errors += 1
        else:
            used = payload.get("usage", {}).get("input_tokens")
            state_tokens = (
                int(used) - overhead if isinstance(used, int) and overhead is not None else None
            )
            answer = payload.get("answers", {}).get("score", {})
            record.update(
                {
                    "state_tokens": state_tokens,
                    "budget_ok": (
                        state_tokens <= BUDGET_TOKENS if state_tokens is not None else None
                    ),
                    "ctx_ok": True,
                    **score_from_answer(answer if isinstance(answer, dict) else {}),
                    "error": None,
                }
            )
            if record["score_continuous"] is not None:
                n_ok += 1
            else:
                n_unscored += 1
        if record["budget_ok"]:
            n_budget += 1
        records.append(record)
        if (index + 1) % 100 == 0:
            print(f"  {model}: {index + 1}/{len(cv_list)}", file=sys.stderr)
    wall = time.perf_counter() - started
    state_counts = [r["state_tokens"] for r in records if r["state_tokens"]]
    disc = [r["score_discrete"] for r in records if r["score_discrete"] is not None]
    cont = [r["score_continuous"] for r in records if r["score_continuous"] is not None]
    return {
        "model": model,
        "q_overhead_tokens": overhead,
        "n_total": len(cv_list),
        "n_ok": n_ok,
        "n_ctx_errors": n_ctx_errors,
        "n_request_errors": n_request_errors,
        "n_unscored": n_unscored,
        "n_budget_ok": n_budget,
        "max_state_tokens": max(state_counts) if state_counts else None,
        "wall_s": round(wall, 1),
        "score_discrete_mean": round(sum(disc) / len(disc), 1) if disc else None,
        "score_continuous_mean": round(sum(cont) / len(cont), 1) if cont else None,
        "records": records,
    }


def sweep_fit(
    model: str,
    states: list[str],
    questions: dict[str, Any],
    *,
    host: str,
    timeout: float,
) -> dict[str, Any]:
    """Sweep every corpus prompt against one model; report context fit.

    The default protocol only exercises ``cvs[0]`` — this phase is what
    ``--sweep`` adds so corpus-fit claims in the docs are reproducible.
    """
    ok = 0
    errors: dict[str, int] = {}
    tokens: list[float] = []
    started = time.perf_counter()
    for state in states:
        _, payload = post_systemone(model, state, questions, host=host, timeout=timeout)
        if "error" in payload:
            message = str(payload["error"])
            errors[message] = errors.get(message, 0) + 1
        else:
            ok += 1
            used = payload.get("usage", {}).get("input_tokens")
            if isinstance(used, int):
                tokens.append(float(used))
    return {
        "total": len(states),
        "ok": ok,
        "rejected": len(states) - ok,
        "wall_s": round(time.perf_counter() - started, 1),
        "errors": errors,
        "input_tokens": latency_stats(tokens),
    }


def bench_model(
    model: str,
    *,
    host: str,
    n: int,
    threads: int,
    timeout: float,
    tags: dict[str, dict[str, Any]],
    sweep_states: list[str] | None = None,
) -> dict[str, Any]:
    """Full protocol for one model: cold, warm short, warm CV, throughput."""
    cv_state = real_cv_state()
    print(f"\n=== {model} ===", file=sys.stderr)
    stop_model(model, host)
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
    stop_model(model, host)

    tag_info: dict[str, Any] = tags.get(model, {})
    if not tag_info and ":" not in model:
        tag_info = tags.get(f"{model}:latest", {})
    corpus_sweep = None
    if sweep_states is not None:
        print(f"  corpus sweep x{len(sweep_states)}...", file=sys.stderr)
        corpus_sweep = sweep_fit(
            model,
            sweep_states,
            build_cv_questions(),
            host=host,
            timeout=timeout,
        )
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
        "corpus_sweep": corpus_sweep,
    }


def print_table(results: list[dict[str, Any]]) -> None:
    hdr = (
        f"{'model':<40} {'ctx':>4} {'cold':>7} {'short p50':>9} {'short p95':>9} "
        f"{'cv p50':>7} {'cv p95':>7} {'in_tok':>6} {'rps_ok':>7} {'cv_err':>6}"
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
            f"{cv_p50:>7} {cv_p95:>7} {in_tok_s:>6} {r['throughput']['rps_ok']:>7.1f} "
            f"{cv_errs:>6}"
        )
        if r["cold_error"]:
            print(f"    cold error: {r['cold_error'][:120]}")
        for message in list(r["cv"]["errors"])[:1]:
            print(f"    cv error: {message[:160]}")
        if r["cv"]["sample_answers"]:
            print(f"    cv answers: {r['cv']['sample_answers']}")
        sweep = r.get("corpus_sweep")
        if sweep:
            tok = sweep["input_tokens"]
            print(
                f"    corpus fit: {sweep['ok']}/{sweep['total']} ok, "
                f"{sweep['rejected']} rejected "
                f"(input_tokens p50={tok['p50']:.0f} max={tok['max']:.0f})"
            )


def print_sweep_summary(results: list[dict[str, Any]]) -> None:
    hdr = (
        f"{'model':<40} {'ok':>5} {'ctx_err':>7} {'other':>5} {'noscore':>7} "
        f"{'<=500':>6} {'max_state':>9} {'cont_mean':>9} {'wall':>7}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        max_state = r["max_state_tokens"]
        print(
            f"{r['model']:<40} {r['n_ok']:>5} {r['n_ctx_errors']:>7} "
            f"{r['n_request_errors']:>5} {r['n_unscored']:>7} "
            f"{r['n_budget_ok']:>6} {max_state!s:>9} "
            f"{r['score_continuous_mean']!s:>9} {r['wall_s']:>6.1f}s"
        )


def run_score_sweep(args: argparse.Namespace) -> None:
    """v2-instrument flow: score every corpus CV per tag (separate protocol)."""
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cv_list = list(cvs)
    tags = fetch_tags(args.host)
    started = datetime.now(UTC).isoformat(timespec="seconds")

    results = [
        score_sweep_model(
            model,
            cv_list,
            host=args.host,
            timeout=args.timeout,
        )
        for model in models
    ]
    report = {
        "meta": {
            "instrument": "decision-v2",
            "budget_tokens": BUDGET_TOKENS,
            "score_levels": 11,
            "host": args.host,
            "n_cvs": len(cv_list),
            "started": started,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.platform(),
            "model_digests": {m: tags.get(m, tags.get(f"{m}:latest", {})) for m in models},
        },
        "models": results,
    }
    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"wrote {args.json_path}", file=sys.stderr)
    print_sweep_summary(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated tags")
    parser.add_argument("--n", type=int, default=50, help="requests per phase")
    parser.add_argument("--threads", type=int, default=4, help="throughput workers")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also sweep all 600 corpus prompts per model (context-fit rates)",
    )
    parser.add_argument(
        "--score-sweep",
        action="store_true",
        help="run only the v2 instrument sweep: score all 600 CVs per model "
        "(decision_state + 11-level score question)",
    )
    parser.add_argument("--json", dest="json_path", help="write full results JSON here")
    args = parser.parse_args()

    if args.score_sweep:
        run_score_sweep(args)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tags = fetch_tags(args.host)
    sweep_states = [real_cv_state_cv(cv) for cv in cvs] if args.sweep else None
    started = datetime.now(UTC).isoformat(timespec="seconds")

    results = [
        bench_model(
            model,
            host=args.host,
            n=args.n,
            threads=args.threads,
            timeout=args.timeout,
            tags=tags,
            sweep_states=sweep_states,
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
