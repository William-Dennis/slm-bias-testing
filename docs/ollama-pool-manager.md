# Design Spec: Ollama Pool Manager

## Goal

Replace Python-side concurrency with a Node.js worker pool that manages all
Ollama API calls. Python stays sequential and simple. Node.js handles
parallelism.

## Architecture

```text
Python (benchmark logic)          Node.js (ollama_pool.mjs)
─────────────────────             ─────────────────────────
1. Generate jobs                  1. Read jobs from stdin
2. Write JSONL to stdin ──────►  2. Dispatch to worker pool
3. Read results from stdout ◄──  3. POST /api/chat to Ollama
4. Process results               4. Write JSONL to stdout
```

## Components

### 1. `scripts/ollama_pool.mjs` — Worker Pool Manager

Responsibilities:
- Read JSONL jobs from stdin (one per line)
- Maintain a pool of N concurrent workers
- Each worker sends POST requests to Ollama `/api/chat`
- Write JSONL results to stdout
- Handle timeouts, retries, and Ollama crashes
- Graceful shutdown on SIGTERM/SIGINT

CLI:
```bash
node scripts/ollama_pool.mjs [options]

Options:
  --max-pool <n>       Upper bound on concurrent workers (default: 6)
  --min-pool <n>       Lower bound — always allow at least this many (default: 1)
  --adaptive           Enable adaptive concurrency (default: true)
  --ram-floor <gb>     Minimum free RAM to maintain (default: 2)
  --latency-ceiling <ms>  Pause if avg latency exceeds this (default: 15000)
  --timeout <ms>       Per-request timeout (default: 30000)
  --retries <n>        Retry count per request (default: 2)
  --ollama-host <url>  Ollama base URL (default: http://localhost:11434)
```

### 2. Python Side — Sequential Benchmark Runner

Changes to existing code:
- Remove ThreadPoolExecutor from `benchmark.py`
- Remove threading imports and locks
- `run_benchmark()` becomes purely sequential
- No changes to benchmark evaluation logic (stereoset, winobias, demographic)

New flow:
- Python opens `ollama_pool.mjs` as subprocess
- Writes jobs to its stdin as they're generated
- Reads results from its stdout
- Processes results in order

## Protocol

### Job Format (Python → Node.js, stdin JSONL)

```json
{
  "id": "stereoset_042_stereotype",
  "model": "smollm:135m",
  "prompt": "Context: The doctor...\nContinuation: He is...",
  "temperature": 0.0,
  "num_ctx": 2048,
  "keep_alive": 5
}
```

Fields:
- `id` (string, required): Unique job identifier. Python uses this to match results.
- `model` (string, required): Ollama model tag.
- `prompt` (string, required): The prompt to send.
- `temperature` (number, optional): Default 0.0.
- `num_ctx` (int, optional): Context window size. Default 2048.
- `keep_alive` (float, optional): Seconds to keep model loaded. Default 5.

### Result Format (Node.js → Python, stdout JSONL)

```json
{
  "id": "stereoset_042_stereotype",
  "response": "85",
  "error": null,
  "latency_ms": 1234
}
```

Fields:
- `id` (string): Matches the job ID.
- `response` (string|null): Ollama response text. Null if error.
- `error` (string|null): Error message if failed. Null if success.
- `latency_ms` (int): Round-trip time in milliseconds.

### Flow Control

- Python writes all jobs for a benchmark, then closes stdin (signals EOF)
- Node.js processes remaining jobs, writes results, exits
- Python reads results until Node.js stdout closes
- If Python crashes, Node.js detects stdin close and drains + exits

## Ollama API

Endpoint: `POST {ollama_host}/api/chat`

Request:
```json
{
  "model": "smollm:135m",
  "messages": [{"role": "user", "content": "..."}],
  "stream": false,
  "options": {
    "temperature": 0.0,
    "num_ctx": 2048
  },
  "keep_alive": 5
}
```

Response (relevant fields):
```json
{
  "message": {"role": "assistant", "content": "..."},
  "done": true,
  "total_duration": 370148000
}
```

## Worker Pool Logic

```text
max_pool = N  (e.g. 6)
active_workers = 0

for each job from stdin:
  wait until active_workers < max_pool AND system has headroom
  active_workers++
  worker sends POST /api/chat
  on success: write result to stdout
  on timeout/error:
    retry up to --retries times
    if still failing: write error result to stdout
  active_workers--
```

Key behaviors:
- Workers are async — up to N requests in flight simultaneously
- Each worker handles one request at a time
- Pool processes jobs in order (FIFO from stdin)
- No batching — jobs stream through as fast as Ollama can handle
- Adaptive concurrency: scales up/down based on system resources

## Adaptive Concurrency

The pool doesn't just use a fixed size — it monitors system resources
and adjusts how many workers are active. This prevents OOM and keeps
throughput optimal.

### Resource Monitoring

 polled every 2 seconds via a background timer:

| Metric        | Source               | No sudo? |
|---------------|----------------------|----------|
| RAM free      | `os.freemem()`       | Yes      |
| RAM total     | `os.totalmem()`      | Yes      |
| CPU load      | `os.loadavg()`       | Yes      |
| CPU cores     | `os.cpus().length`   | Yes      |
| Ollama latency| rolling avg of last 20 responses | Yes |

GPU usage can't be read on macOS without sudo. Instead, we use
Ollama response latency as a proxy — high latency = GPU saturated.

### Headroom Rules

```text
can_dispatch = (
  free_ram > RAM_FLOOR           # e.g. 2GB always free
  AND cpu_load < cores * 0.9     # don't saturate CPU
  AND avg_latency < LATENCY_CEIL # e.g. 10s — GPU not thrashed
)
```

If `can_dispatch` is false, the dispatcher pauses until resources
free up. Workers already in flight continue to completion.

### Scaling Behavior

| Condition                     | Action                          |
|-------------------------------|---------------------------------|
| free_ram > 4GB, low latency   | Allow up to max_pool workers    |
| free_ram 2-4GB, or latency up | Cap at max_pool / 2 workers     |
| free_ram < 2GB, or latency >20s | Pause dispatch, drain to min_pool   |
| Ollama connection refused     | Pause all, retry connection     |

### CLI Flags

```bash
--adaptive          Enable adaptive concurrency (default: true)
--max-pool <n>      Upper bound on concurrent workers (default: 6)
--min-pool <n>      Lower bound — always allow at least this many (default: 1)
--ram-floor <gb>    Minimum free RAM to maintain (default: 2)
--latency-ceiling <ms>  Pause if avg latency exceeds this (default: 15000)
```

### Monitoring Output

Every 30 seconds, write a status line to stderr:

```text
[pool] workers: 4/6  ram: 18.2/24GB  cpu: 3.2/10  latency: 2.1s  dispatched: 1247  errors: 3
```

This gives real-time visibility without cluttering the JSONL protocol.

## Concurrency Tuning

The max-pool size sets the upper bound. Adaptive logic scales down
when resources are tight.

| max-pool | OLLAMA_NUM_PARALLEL | RAM headroom | Use case |
|----------|---------------------|--------------|----------|
| 2        | 2                   | 4GB+         | Safe, low memory |
| 4        | 4                   | 3GB+         | Balanced |
| 6        | 6                   | 2GB+         | Aggressive |
| 8        | 8                   | 2GB+         | Max throughput, 16GB+ |

With adaptive mode, the pool auto-scales within these bounds based
on actual system load.

## Error Handling

### Node.js Side
- Connection refused → retry with backoff (1s, 2s, 4s)
- Timeout → retry with same backoff
- All retries exhausted → write error result
- Ollama returns non-200 → write error result
- Stdin closed → drain in-flight jobs, write results, exit

### Python Side
- Error result → skip that data point (same as current fallback behavior)
- Node.js crashes → detect via subprocess exit, restart if needed
- Partial results → checkpoint saved incrementally (existing behavior)

## Files to Create/Modify

### Create
- `scripts/ollama_pool.mjs` (~150 lines)

### Modify
- `src/slm_bias_testing/benchmark.py` — remove ThreadPoolExecutor, use pool
- `src/slm_bias_testing/call_api.py` — add pool-based Model class
- `scripts/run_single_model.py` — pass pool size, no threading flags
- `scripts/run_parallel.py` — remove threading flags

### No Changes
- `stereoset.py`, `winobias.py`, `demographic_bias.py` — evaluation logic unchanged
- `registry.py`, `analysis.py`, `temporal.py`, `visualisations.py`

## Estimated Impact

Current (Python threading, OLLAMA_NUM_PARALLEL=4):
- 33 calls/min → ~40 hours

With pool_size=4 (same parallelism, but actually used everywhere):
- ~100 calls/min → ~13 hours

With pool_size=6:
- ~150 calls/min → ~9 hours

With pool_size=8:
- ~200 calls/min → ~7 hours

The gain comes from stereoset/winoBias/demographic benchmarks now
using the pool instead of sequential calls.

## Verification

1. Build ollama_pool.mjs, test standalone:
   echo '{"id":"test","model":"smollm:135m","prompt":"Say hi"}' | node scripts/ollama_pool.mjs --max-pool 2

2. Test adaptive mode — watch stderr for resource monitoring:
   echo '{"id":"test","model":"smollm:135m","prompt":"Say hi"}' | node scripts/ollama_pool.mjs --adaptive --max-pool 4 2>&1

3. Test Python integration with max-samples 5:
   uv run python scripts/run_single_model.py smollm-135m --max-samples 5

4. Test parallel with 2 models:
   uv run python scripts/run_parallel.py --concurrency 2 --models smollm-135m,gemma3-270m

5. Verify results match previous run format

6. Full overnight run with max-pool 6, adaptive
