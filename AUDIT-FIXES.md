# Audit Fix List — slm-bias-testing

Fix every item below. One pass, all files. Run `uv run ruff check src tests && uv run ruff format src tests && uv run mypy src/slm_bias_testing && uv run pytest -q` after all changes to verify.

## CRITICAL

### 1. cv-screening checkpoint (benchmark.py)
The CV screening benchmark (`src/slm_bias_testing/benchmark.py`) saves results only at the end via `save_records()`. If the process is killed, ALL results are lost. Add per-call checkpointing using the same pattern as the other benchmarks:
- After each `process_cv_run()` returns a record, append it to a JSONL checkpoint file
- On resume, load the checkpoint and skip already-completed (key, run) pairs
- The checkpoint file should be `records_checkpoint.jsonl` in the output_dir

### 2. Narrow except Exception in call_api.py
`_predict_ollama()` catches bare `except Exception` which swallows KeyboardInterrupt. Change to:
```python
except (ollama.ResponseError, ConnectionError, TimeoutError, OSError) as e:
```
Keep the existing retry logic but only catch retryable errors. Let KeyboardInterrupt, SystemExit, etc. propagate.

### 3. Atomic results.json write in runner.py
Line 146-147: `json.dump` to results.json is not atomic. If it fails mid-write, the corrupt file causes permanent skip on retry. Fix: write to a temp file first, then `os.rename()`:
```python
import tempfile
tmp_path = results_file + ".tmp"
with open(tmp_path, "w") as f:
    json.dump(summary, f, indent=2)
os.rename(tmp_path, results_file)
```

### 4. Overnight script branch protection (scripts/overnight_benchmark.sh)
Lines 108-116: The script commits and pushes directly to `origin/main`, violating the repo rule "Never commit directly to main." Fix: create a branch, commit there, then push. Change the git section to:
```bash
BRANCH="results/$(date +%Y-%m-%d)"
git checkout -b "$BRANCH"
git add "results/*/*/results.json"
if ! git diff --cached --quiet; then
    git commit -m "chore: overnight benchmark results $(date +%Y-%m-%d)
Results: $completed new, $skipped skipped, $failed failed"
    git push origin "$BRANCH"
    echo "Pushed to branch $BRANCH — create a PR to merge."
else
    echo "No new results to commit"
fi
git checkout main
```
Remove the `git push origin main` line.

## HIGH

### 5. Default score 50 warning (stereoset.py)
`_score_sentence()` returns 50 on failure (line 43-45). This is indistinguishable from a real score. Add a warning log when the default is used:
```python
except Exception:
    logger.exception("Failed to score continuation")
    logger.warning("Using default score 50 for: %s / %s", context[:50], continuation[:50])
```

### 6. Per-call latency logging (call_api.py)
Add timing to `_predict_ollama()`. Log the duration of each successful call at DEBUG level:
```python
import time
start = time.monotonic()
# ... existing chat call ...
elapsed = time.monotonic() - start
logger.debug("Ollama call completed in %.2fs (attempt %d)", elapsed, attempt + 1)
```

### 7. OllamaClient.ensure_running() silent failure (call_api.py)
Line 45: if auto-restart also fails, the method returns silently. Log at ERROR level when restart fails:
```python
except Exception:
    logger.error("Ollama auto-restart failed — subsequent API calls will fail", exc_info=True)
    raise
```
Actually, just let the exception propagate instead of catching it silently.

### 8. Corrupted results logging (visualisations.py, temporal.py)
Both files silently skip JSONDecodeError/OSError when loading results. Add a warning log:
```python
except (json.JSONDecodeError, OSError):
    logger.warning("Skipping corrupted results file: %s", path)
    continue
```

### 9. pkill overly broad (ollama_setup.py)
Line 36: `pkill -f "ollama serve"` could kill unrelated processes. Make it more specific by targeting the PID instead. Since `OllamaServer` tracks `self.process`, we can use that. But the `_kill_existing_ollama` runs before start, so we don't have a PID yet. Instead, use a more targeted pattern:
```python
subprocess.run(
    ["pkill", "-f", "ollama serve --"],
    ...
)
```
The `--` helps prevent matching unrelated commands. Or better: log a warning that existing processes may be killed.

## MEDIUM

### 10. Verbose flag for logging (runner.py)
Line 175: `logging.basicConfig(level=logging.INFO)` is hardcoded. Add a `--verbose` flag:
```python
parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
```
And in main():
```python
logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
```

### 11. Skip events at INFO (stereoset.py, winobias.py)
Lines that log at DEBUG for skipped items should be at INFO so operators can see them:
```python
logger.info("Skipping item %s: missing stereotype or anti-stereotype", item.get("id"))
```
Same for winobias skip logging.

### 12. Exponential backoff in retry (call_api.py)
Line 117: constant `time.sleep(2)`. Change to exponential backoff:
```python
sleep_time = min(2 ** attempt, 30)  # 2s, 4s, 8s — capped at 30s
time.sleep(sleep_time)
```

### 13. ollama pull timeout (runner.py)
Lines 29-30: `subprocess.run(["ollama", "pull", ...])` has no timeout. Add one:
```python
result = subprocess.run(
    ["ollama", "pull", ollama_tag],
    capture_output=True,
    text=True,
    timeout=600,  # 10 minutes
)
```

### 14. NaN filtering in temporal.py
Line 161: `sp_stats.linregress` with NaN produces NaN results. Filter NaN before regression:
```python
mask = ~(np.isnan(x) | np.isnan(y))
if mask.sum() >= 3:
    slope, intercept, r_val, p_val, _std_err = sp_stats.linregress(x[mask], y[mask])
```

### 15. Checkpoint cleanup (benchmarks/__init__.py)
After `save_results()` writes the final results.json, delete the checkpoint file:
```python
def save_results(self, results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{self.name}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    # Clean up checkpoint
    ckpt = self._checkpoint_path(output_dir)
    if os.path.exists(ckpt):
        os.remove(ckpt)
```

## LOW

### 16. Unused timeout parameter (benchmark.py)
Line 155: `timeout` parameter is documented as "Unused for now". Remove it from the signature and update callers. Or add a `# TODO` comment. I prefer removing it since it's dead code.

### 17. frozenset for entity matching (winobias.py)
Line 58-74: `_find_entities()` uses tuple `("the", "a", "an")` for membership checks. Change to frozenset for O(1) lookups:
```python
_ARTICLES = frozenset({"the", "a", "an"})
```
Use it in the while loop condition.

### 18. Python version pin (pyproject.toml)
Line 9: `requires-python = ">=3.11"` is unpinned. Change to:
```toml
requires-python = ">=3.11,<3.14"
```

After ALL changes, run the full verification:
```
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src/slm_bias_testing && uv run pytest -q
```
