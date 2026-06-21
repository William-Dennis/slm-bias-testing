#!/usr/bin/env python3
"""End-to-end benchmark for the Ollama pool.

Tests pool throughput at different concurrency levels and compares
against raw Ollama API calls (no pool harness).

Usage:
    uv run python scripts/benchmark_pool.py
    uv run python scripts/benchmark_pool.py --model smollm:135m --jobs 20
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"


def _check_ollama() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5):
            return True
    except Exception:
        return False


def _ollama_chat(prompt: str, model: str) -> float:
    """Single Ollama API call. Returns latency in seconds."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 2048},
            "keep_alive": 30,
        }
    ).encode()
    start = time.monotonic()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=60)
    resp.read()
    resp.close()
    return time.monotonic() - start


def _pool_batch(jobs: list[dict], pool_size: int, no_restart: bool = False) -> list[dict]:
    """Run jobs through the pool. Returns list of result dicts."""
    script = str(Path(__file__).resolve().parent.parent / "scripts" / "ollama_pool.mjs")
    cmd = [
        "node",
        script,
        "--max-pool",
        str(pool_size),
        "--no-adaptive",
        "--ollama-host",
        OLLAMA_HOST,
    ]
    if no_restart:
        cmd.append("--no-restart")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    stdin = proc.stdin
    stdout = proc.stdout
    stderr = proc.stderr
    stderr_lines: list[str] = []

    def drain():
        for line in stderr:
            stderr_lines.append(line.rstrip())

    t = threading.Thread(target=drain, daemon=True)
    t.start()

    # Read and validate handshake (first line from pool)
    handshake_line = stdout.readline()
    if not handshake_line:
        proc.terminate()
        raise RuntimeError("Pool closed before sending handshake")
    try:
        handshake = json.loads(handshake_line)
        if not isinstance(handshake, dict) or handshake.get("protocol") != 1 or not handshake.get("ready"):
            proc.terminate()
            raise RuntimeError(f"Pool sent unexpected handshake: {handshake}")
    except json.JSONDecodeError:
        proc.terminate()
        raise RuntimeError(f"Pool sent invalid handshake: {handshake_line!r}")

    for job in jobs:
        stdin.write(json.dumps(job) + "\n")
    stdin.flush()
    with contextlib.suppress(OSError):
        stdin.close()

    results = []
    for _ in range(len(jobs)):
        line = stdout.readline()
        if not line:
            break
        results.append(json.loads(line))

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    t.join(timeout=5)
    return results


def bench_raw_sequential(n: int, model: str) -> dict:
    """Baseline: raw Ollama API, one call at a time."""
    latencies = []
    for i in range(n):
        latencies.append(_ollama_chat(f"Say hello number {i}", model))
    total = sum(latencies)
    return {
        "test": "raw_sequential",
        "workers": 1,
        "jobs": n,
        "total_s": round(total, 2),
        "throughput": round(n / total, 1),
        "avg_latency_s": round(total / n, 2),
    }


def bench_pool(pool_size: int, n: int, model: str, no_restart: bool = False) -> dict:
    """Pool benchmark at given concurrency."""
    jobs = [
        {"id": f"b_{i}", "model": model, "prompt": f"Say hello number {i}", "temperature": 0.0}
        for i in range(n)
    ]
    start = time.monotonic()
    results = _pool_batch(jobs, pool_size, no_restart=no_restart)
    total = time.monotonic() - start
    errors = sum(1 for r in results if r.get("error"))
    latencies = [r["latency_ms"] / 1000 for r in results if not r.get("error")]
    return {
        "test": f"pool_{pool_size}w",
        "workers": pool_size,
        "jobs": n,
        "completed": len(results) - errors,
        "errors": errors,
        "total_s": round(total, 2),
        "throughput": round(len(results) / total, 1),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Ollama pool throughput")
    parser.add_argument("--model", default="smollm:135m")
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()

    if not _check_ollama():
        print("ERROR: Ollama not running", file=sys.stderr)
        sys.exit(1)

    n, model = args.jobs, args.model
    print(f"Benchmarking: {n} jobs, model={model}\n")
    results = []

    print("Running: raw sequential (1 worker)...", flush=True)
    r = bench_raw_sequential(n, model)
    results.append(r)
    print(f"  {r['throughput']} calls/sec, {r['total_s']}s total")

    for w in [1, 2, 3, 4]:
        print(f"Running: pool {w} workers...", flush=True)
        r = bench_pool(w, n, model, no_restart=True)
        results.append(r)
        err = f", {r['errors']} errors" if r["errors"] else ""
        print(f"  {r['throughput']} calls/sec, {r['total_s']}s total{err}")

    baseline = results[0]["throughput"]
    print(f"\n{'=' * 60}")
    print(f"  {'TEST':<20s} {'WORKERS':>7s} {'THROUGHPUT':>12s} {'TOTAL':>8s} {'SPEEDUP':>8s}")
    print(f"  {'-' * 20} {'-' * 7} {'-' * 12} {'-' * 8} {'-' * 8}")
    for r in results:
        speedup = r["throughput"] / baseline if baseline else 0
        print(
            f"  {r['test']:<20s} {r['workers']:>7d} {r['throughput']:>9.1f}/s "
            f"{r['total_s']:>7.1f}s {speedup:>7.1f}x"
        )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
