"""Pick (max_batch_size, max_latency_ms) once, then freeze.

For each cell of a small grid, restart BentoML with that batching config and
drive an open-loop load (matches the proposal's anti-Coordinated-Omission
requirement). Collect P50/P95/P99 latency and error count. Pick the cell with
the lowest P99 among zero-error cells; tie-break on P50.

The picked config is written to batching.json. A markdown report is written
to calibration_report.md for the thesis appendix.

Usage:
    uv run python calibrate_batching.py
    uv run python calibrate_batching.py --target-rps 200 --duration 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# project/ — two levels above project/2-serving/bentoml/
PROJECT_ROOT = ROOT.parents[1]
ARTIFACTS = PROJECT_ROOT / "1-model" / "artifacts"
TEST_PARQUET = PROJECT_ROOT / "1-model" / "data" / "test.parquet"
BATCHING_PATH = ROOT / "batching.json"
REPORT_PATH = ROOT / "calibration_report.md"

PORT = 8099
URL = f"http://127.0.0.1:{PORT}"

GRID_BATCH = [16, 32, 64]
GRID_LATENCY_MS = [2, 5, 10]
WARMUP_REQUESTS = 100
READY_TIMEOUT_S = 30


def write_batching(max_batch_size: int, max_latency_ms: int, note: str) -> None:
    BATCHING_PATH.write_text(
        json.dumps(
            {"max_batch_size": max_batch_size, "max_latency_ms": max_latency_ms, "_note": note},
            indent=2,
        ),
        encoding="utf-8",
    )


def sample_row() -> list[float]:
    df = pd.read_parquet(TEST_PARQUET)
    feature_cols = [c for c in df.columns if c != "label"]
    row = df.sample(n=1, random_state=int(time.time()))[feature_cols].iloc[0].tolist()
    return [float(x) for x in row]


def start_server(env: dict[str, str], log_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "bentoml", "serve", "service:TgService",
        "--host", "127.0.0.1", "--port", str(PORT),
    ]
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env={**os.environ, **env},
        stdout=log_fh, stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]  # keep handle alive
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    log_fh = getattr(proc, "_log_fh", None)
    try:
        if proc.poll() is None:
            if os.name == "nt":
                # Walk the child tree — BentoML spawns worker subprocesses that
                # CTRL_BREAK doesn't reliably reach. taskkill /T handles this.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                )
            else:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    finally:
        if log_fh is not None:
            log_fh.close()


async def wait_ready(client: httpx.AsyncClient, row: list[float]) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            r = await client.post(f"{URL}/predict", json={"features": [row]}, timeout=2.0)
            if r.status_code == 200:
                return
        except (httpx.HTTPError, httpx.ConnectError):
            pass
        await asyncio.sleep(0.5)
    raise TimeoutError(f"server not ready after {READY_TIMEOUT_S}s")


async def warmup(client: httpx.AsyncClient, row: list[float], n: int) -> None:
    for _ in range(n):
        await client.post(f"{URL}/predict", json={"features": [row]}, timeout=5.0)


async def fire_one(client: httpx.AsyncClient, row: list[float]) -> tuple[float, bool]:
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{URL}/predict", json={"features": [row]}, timeout=10.0)
        ok = r.status_code == 200
    except Exception:
        ok = False
    return time.perf_counter() - t0, ok


async def open_loop_load(
    client: httpx.AsyncClient, row: list[float], target_rps: int, duration_s: float,
) -> tuple[list[float], int]:
    interval = 1.0 / target_rps
    tasks: list[asyncio.Task] = []
    start = time.monotonic()
    next_fire = start
    while True:
        now = time.monotonic()
        if now - start >= duration_s:
            break
        if now >= next_fire:
            tasks.append(asyncio.create_task(fire_one(client, row)))
            next_fire += interval
        else:
            await asyncio.sleep(max(0.0, next_fire - now))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    latencies: list[float] = []
    errors = 0
    for r in results:
        if isinstance(r, BaseException):
            errors += 1
            continue
        lat, ok = r
        if not ok:
            errors += 1
        else:
            latencies.append(lat * 1000.0)  # ms
    return latencies, errors


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(xs, p))


async def run_one_cell(
    max_batch_size: int, max_latency_ms: int, target_rps: int, duration_s: float,
) -> dict:
    note = f"calibration cell batch={max_batch_size} latency_ms={max_latency_ms}"
    write_batching(max_batch_size, max_latency_ms, note)

    env = {
        "WEB_CONCURRENCY": "1",
        "ORT_INTRA_OP_NUM_THREADS": "2",
        "ORT_INTER_OP_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "MODEL_PATH": str(ARTIFACTS / "model.onnx"),
        "SCHEMA_PATH": str(ARTIFACTS / "schema.json"),
        "BATCHING_PATH": str(BATCHING_PATH),
    }
    log_path = ROOT / f".calib_b{max_batch_size}_l{max_latency_ms}.log"
    print(f"[calib] cell batch={max_batch_size} latency_ms={max_latency_ms}: starting server...")
    proc = start_server(env, log_path)

    row = sample_row()
    try:
        async with httpx.AsyncClient() as client:
            await wait_ready(client, row)
            await warmup(client, row, WARMUP_REQUESTS)
            t0 = time.monotonic()
            latencies, errors = await open_loop_load(client, row, target_rps, duration_s)
            elapsed = time.monotonic() - t0
    finally:
        stop_server(proc)

    summary = {
        "max_batch_size": max_batch_size,
        "max_latency_ms": max_latency_ms,
        "n": len(latencies),
        "errors": errors,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "p999_ms": percentile(latencies, 99.9),
        "mean_ms": statistics.fmean(latencies) if latencies else float("nan"),
        "achieved_rps": len(latencies) / elapsed if elapsed > 0 else 0.0,
        "elapsed_s": elapsed,
    }
    print(
        f"[calib]   n={summary['n']} err={summary['errors']} "
        f"P50={summary['p50_ms']:.2f} P95={summary['p95_ms']:.2f} "
        f"P99={summary['p99_ms']:.2f} achieved_rps={summary['achieved_rps']:.1f}"
    )
    return summary


def pick_best(results: list[dict]) -> dict:
    clean = [r for r in results if r["errors"] == 0]
    pool = clean if clean else results
    return min(pool, key=lambda r: (r["p99_ms"], r["p50_ms"]))


def write_report(results: list[dict], best: dict, target_rps: int, duration_s: float) -> None:
    lines = [
        "# BentoML Adaptive Batching Calibration",
        "",
        f"Target load: {target_rps} RPS open-loop, {duration_s:.0f}s per cell, single-row requests.",
        f"Fixed parallelism: WEB_CONCURRENCY=1, ORT_INTRA_OP_NUM_THREADS=2 (the most batching-friendly cell of the Phase 2 matrix).",
        "",
        "| max_batch_size | max_latency_ms | N | errors | P50 ms | P95 ms | P99 ms | P99.9 ms | achieved RPS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        marker = " **(picked)**" if (r["max_batch_size"] == best["max_batch_size"]
                                     and r["max_latency_ms"] == best["max_latency_ms"]) else ""
        lines.append(
            f"| {r['max_batch_size']} | {r['max_latency_ms']} | {r['n']} | {r['errors']} | "
            f"{r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | {r['p999_ms']:.2f} | "
            f"{r['achieved_rps']:.1f} |{marker}"
        )
    lines += [
        "",
        f"**Picked**: max_batch_size={best['max_batch_size']}, "
        f"max_latency_ms={best['max_latency_ms']} (lowest P99 with zero errors; tie-break on P50).",
        "",
        "This pair is written into [batching.json](batching.json) and held fixed across the "
        "Phase 3 worker × ORT-thread matrix.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[calib] report -> {REPORT_PATH}")


async def main_async(args) -> int:
    results: list[dict] = []
    for batch in GRID_BATCH:
        for lat in GRID_LATENCY_MS:
            r = await run_one_cell(batch, lat, args.target_rps, args.duration)
            results.append(r)

    best = pick_best(results)
    write_batching(best["max_batch_size"], best["max_latency_ms"],
                   "calibrated; see calibration_report.md")
    write_report(results, best, args.target_rps, args.duration)
    print(
        f"[calib] DONE: batch={best['max_batch_size']} "
        f"latency_ms={best['max_latency_ms']} P99={best['p99_ms']:.2f}ms"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-rps", type=int, default=100)
    p.add_argument("--duration", type=float, default=20.0, help="seconds per cell")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
