#!/usr/bin/env python3
"""Run all benchmarks for one or more models using the Node.js Ollama pool.

Usage:
    uv run python scripts/run_benchmarks.py --models smollm-135m,gemma3-270m --pool-size 4
    uv run python scripts/run_benchmarks.py --models all --pool-size 6
    uv run python scripts/run_benchmarks.py --models smollm-135m --benchmark stereoset
"""

from slm_bias_testing.benchmark_runner import main

if __name__ == "__main__":
    main()
