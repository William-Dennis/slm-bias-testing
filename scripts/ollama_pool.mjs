#!/usr/bin/env node
/**
 * Ollama Pool Manager — worker pool for parallel Ollama API calls.
 *
 * Reads JSONL jobs from stdin, dispatches to Ollama /api/chat concurrently,
 * writes JSONL results to stdout.
 *
 * Usage:
 *   echo '{"id":"test","model":"smollm:135m","prompt":"Say hi"}' | node ollama_pool.mjs --max-pool 4
 *
 * Protocol:
 *   Job:    {"id":"...","model":"...","prompt":"...","temperature":0.0,"num_ctx":2048,"keep_alive":30}
 *   Result: {"id":"...","response":"...","error":null,"latency_ms":1234}
 */

import * as readline from "node:readline";
import * as http from "node:http";
import * as os from "node:os";
import { spawnSync, spawn } from "node:child_process";

// ── CLI args ──────────────────────────────────────────────────────────
const args = process.argv.slice(2);
function flag(name, fallback) {
  const idx = args.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  const val = args[idx + 1];
  if (val === undefined || val.startsWith("--")) return fallback;
  return val;
}

const MAX_POOL = Math.max(1, parseInt(flag("max-pool", "6"), 10));
const MIN_POOL = Math.max(1, Math.min(MAX_POOL, parseInt(flag("min-pool", "1"), 10)));
const ADAPTIVE = !args.includes("--no-adaptive");
const RAM_FLOOR_GB = parseFloat(flag("ram-floor", "2"));
const LATENCY_CEILING_MS = parseInt(flag("latency-ceiling", "15000"), 10);
const TIMEOUT_MS = parseInt(flag("timeout", "30000"), 10);
const RETRIES = Math.max(0, parseInt(flag("retries", "2"), 10));
const OLLAMA_HOST = flag("ollama-host", "http://localhost:11434");
const SHOULD_RESTART = args.includes("--restart");

// ── State ─────────────────────────────────────────────────────────────
let activeWorkers = 0;
let pendingJobs = 0;  // jobs submitted but not yet completed
let dispatched = 0;
let completed = 0;
let errors = 0;
let stdinClosed = false;
let poolClosing = false;
const latencyWindow = [];
const LATENCY_WINDOW_SIZE = 20;
const CPU_CORES = os.cpus().length;

function log(msg) { process.stderr.write(`[pool] ${msg}\n`); }
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function writeResult(id, response, error, latencyMs) {
  process.stdout.write(JSON.stringify({ id, response, error, latency_ms: latencyMs }) + "\n");
}

// ── Ollama lifecycle ──────────────────────────────────────────────────
async function setupOllama() {
  if (!SHOULD_RESTART) {
    if (await checkOllama()) {
      log("Ollama already running");
      return true;
    }
    log("ERROR: Ollama not running");
    return false;
  }

  spawnSync("pkill", ["-f", "ollama serve"], { stdio: "ignore" });
  await sleep(1000);

  const env = { ...process.env, OLLAMA_NUM_PARALLEL: String(MAX_POOL) };
  const child = spawn("ollama", ["serve"], { env, stdio: "ignore", detached: true });
  child.unref();

  for (let i = 0; i < 15; i++) {
    await sleep(1000);
    if (await checkOllama()) {
      log(`Ollama started with OLLAMA_NUM_PARALLEL=${MAX_POOL}`);
      return true;
    }
  }
  log("ERROR: Ollama failed to start");
  return false;
}

async function checkOllama() {
  const url = new URL("/api/tags", OLLAMA_HOST);
  try {
    await httpGet(url);
    return true;
  } catch {
    return false;
  }
}

// ── HTTP helpers ──────────────────────────────────────────────────────
function httpGet(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, { timeout: 5000 }, (res) => {
      res.setEncoding("utf8");
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        if (res.statusCode >= 200 && res.statusCode < 300) resolve(data);
        else reject(new Error(`HTTP ${res.statusCode}`));
      });
    });
    req.on("error", reject);
    req.on("timeout", () => { req.destroy(); reject(new Error("timeout")); });
  });
}

function httpPost(url, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const parsed = new URL(url);
    const options = {
      hostname: parsed.hostname, port: parsed.port, path: parsed.pathname,
      method: "POST",
      headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) },
      timeout: TIMEOUT_MS,
    };
    const req = http.request(options, (res) => {
      res.setEncoding("utf8");
      let responseData = "";
      res.on("data", (chunk) => (responseData += chunk));
      res.on("end", () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          try { resolve(JSON.parse(responseData)); }
          catch { reject(new Error(`Invalid JSON: ${responseData.substring(0, 200)}`)); }
        } else {
          reject(new Error(`HTTP ${res.statusCode}: ${responseData.substring(0, 200)}`));
        }
      });
    });
    req.on("error", reject);
    req.on("timeout", () => { req.destroy(); reject(new Error("timeout")); });
    req.write(data);
    req.end();
  });
}

// ── Adaptive concurrency ──────────────────────────────────────────────
function canDispatch() {
  if (activeWorkers >= MAX_POOL) return false;
  if (activeWorkers < MIN_POOL) return true;
  if (!ADAPTIVE) return true;
  const freeRamGB = os.freemem() / (1024 ** 3);
  const cpuLoad = os.loadavg()[0];
  if (freeRamGB < RAM_FLOOR_GB) return false;
  if (cpuLoad > CPU_CORES * 0.9) return false;
  if (latencyWindow.length >= 5) {
    const avg = latencyWindow.reduce((a, b) => a + b, 0) / latencyWindow.length;
    if (avg > LATENCY_CEILING_MS) return false;
  }
  return true;
}

// ── Worker pool ───────────────────────────────────────────────────────
async function dispatchJob(job) {
  pendingJobs++;
  while (!canDispatch() && !poolClosing) await sleep(50);
  if (poolClosing) { pendingJobs--; return; }

  activeWorkers++;
  dispatched++;
  const startTime = Date.now();

  for (let attempt = 0; attempt <= RETRIES; attempt++) {
    try {
      const url = new URL("/api/chat", OLLAMA_HOST);
      const body = {
        model: job.model,
        messages: [{ role: "user", content: job.prompt }],
        stream: false,
        options: { temperature: job.temperature ?? 0.0, num_ctx: job.num_ctx ?? 2048 },
        keep_alive: job.keep_alive ?? 30,
      };
      const response = await httpPost(url.toString(), body);
      const latencyMs = Date.now() - startTime;
      const content = response?.message?.content ?? "";

      latencyWindow.push(latencyMs);
      if (latencyWindow.length > LATENCY_WINDOW_SIZE) latencyWindow.shift();

      writeResult(job.id, content, null, latencyMs);
      completed++;
      activeWorkers--;
      pendingJobs--;
      return;
    } catch (e) {
      if (attempt < RETRIES) {
        await sleep(Math.min(1000 * Math.pow(2, attempt + 1), 4000));
      }
    }
  }

  writeResult(job.id, null, "max retries exceeded", Date.now() - startTime);
  errors++;
  activeWorkers--;
  pendingJobs--;
}

// ── Status reporting ──────────────────────────────────────────────────
function logStatus() {
  const freeGB = (os.freemem() / (1024 ** 3)).toFixed(1);
  const totalGB = (os.totalmem() / (1024 ** 3)).toFixed(0);
  const cpu = os.loadavg()[0].toFixed(1);
  const lat = latencyWindow.length > 0
    ? (latencyWindow.reduce((a, b) => a + b, 0) / latencyWindow.length / 1000).toFixed(1) + "s"
    : "n/a";
  log(`workers: ${activeWorkers}/${MAX_POOL}  ram: ${freeGB}/${totalGB}GB  cpu: ${cpu}/${CPU_CORES}  latency: ${lat}  dispatched: ${dispatched}  completed: ${completed}  errors: ${errors}`);
}

// ── Main ──────────────────────────────────────────────────────────────
const jobQueue = [];

// Start reading stdin immediately (before Ollama setup)
const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  try {
    const job = JSON.parse(trimmed);
    if (!job.id || !job.model || !job.prompt) {
      log(`WARN: skipping invalid job: ${trimmed.substring(0, 100)}`);
      return;
    }
    jobQueue.push(job);
  } catch {
    log(`WARN: skipping invalid JSON: ${trimmed.substring(0, 100)}`);
  }
});
rl.on("close", () => { stdinClosed = true; log("stdin closed, draining queue..."); });

async function main() {
  // Setup Ollama while stdin is already being read and jobs queued
  if (!(await setupOllama())) process.exit(1);

  const     statusInterval = setInterval(logStatus, 30000);

  // Signal readiness via structured handshake on stdout
  process.stdout.write(JSON.stringify({ protocol: 1, ready: true }) + "\n");

  // Process jobs that arrived during Ollama setup
  while (!poolClosing) {
    if (jobQueue.length > 0) {
      dispatchJob(jobQueue.shift());
    } else if (stdinClosed && pendingJobs === 0) {
      break;
    } else {
      await sleep(50);
    }
  }

  clearInterval(statusInterval);
  logStatus();
  log(`done — ${completed} completed, ${errors} errors`);
  process.exit(0);
}

process.on("SIGTERM", () => { log("SIGTERM received, shutting down..."); poolClosing = true; });
process.on("SIGINT", () => { log("SIGINT received, shutting down..."); poolClosing = true; });

main().catch((e) => { log(`FATAL: ${e.message}`); process.exit(1); });
