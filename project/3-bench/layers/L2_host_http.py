"""Layer 2 — host HTTP (server on the host via uv, no Docker; Vegeta sidecar).

Measures framework + request-parsing + TCP-loopback cost without the server's
Docker layer. Compared against L3 (same cell, containerised) it isolates the
*server-side* Docker overhead; compared against L1 it isolates framework + HTTP.

One cell per pipeline: (workers=1, threads=2) — the cell BentoML's batching was
calibrated on, so the budget attribution is cleanest there.

Caveat: the host server is not strictly core-pinned (ORT intra_op=2 still caps
compute concurrency to 2 threads, but the OS may migrate them across cores).
L2 is a budget-attribution probe at moderate RPS, not a saturation sweep, so
this is acceptable; it is documented in thesis/phase2.md.

Usage:
    uv run python layers/L2_host_http.py
    uv run python layers/L2_host_http.py --rps 100 --duration 60
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "layer2"
TARGETS_ROOT = Path(__file__).resolve().parents[1] / "targets"
BODIES = TARGETS_ROOT / "bodies.jsonl"

SERVING = lib.ROOT / "2-serving"
PORTS = {"fastapi": 8011, "bentoml": 8012}
WARMUP = 100
ZERO_ROW = [0.0] * 28


SERVER_CPUSET = "0,1"   # OS-level pinning when launching the host server on Linux


def server_cmd(pipeline: str, port: int, batching_path: Path | None = None) -> tuple[list[str], Path, dict]:
    env = {
        **os.environ,
        "WEB_CONCURRENCY": "1",
        "ORT_INTRA_OP_NUM_THREADS": "2",
        "ORT_INTER_OP_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "MODEL_PATH": str(lib.ARTIFACTS / "model.onnx"),
        "SCHEMA_PATH": str(lib.ARTIFACTS / "schema.json"),
    }
    if pipeline == "fastapi":
        cwd = SERVING / "fastapi"
        env["FASTAPI_THREADPOOL_SIZE"] = "32"
        cmd = ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0",
               "--port", str(port), "--workers", "1", "--no-access-log"]
    else:
        cwd = SERVING / "bentoml"
        env["BATCHING_PATH"] = str(batching_path) if batching_path else str(cwd / "batching.json")
        cmd = ["uv", "run", "bentoml", "serve", "service:TgService",
               "--host", "0.0.0.0", "--port", str(port)]
    # OS-level CPU pinning via `taskset` when available (Linux). Windows has
    # no equivalent that's easy to wire through; this is the meaningful host
    # quality-of-isolation improvement when running from WSL2.
    if os.name == "posix" and __import__("shutil").which("taskset"):
        cmd = ["taskset", "-c", SERVER_CPUSET] + cmd
    return cmd, cwd, env


def run_pipeline(pipeline: str, rps: int, duration: int,
                 label: str | None = None, batching_path: Path | None = None) -> dict:
    label = label or pipeline
    port = PORTS[pipeline]
    cfg = lib.PIPELINES[pipeline]
    cell_dir = RESULTS / label
    cell_dir.mkdir(parents=True, exist_ok=True)
    log = open(cell_dir / "server.log", "wb")

    cmd, cwd, env = server_cmd(pipeline, port, batching_path=batching_path)
    print(f"[L2] {label}: launching host server on :{port} (BATCHING_PATH={env.get('BATCHING_PATH','-')})")
    # On Linux, start the server in its own process group so we can SIGTERM the
    # whole tree cleanly; on Windows we still use taskkill /F /T further down.
    popen_kwargs: dict = {"cwd": str(cwd), "env": env, "stdout": log, "stderr": subprocess.STDOUT}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        lib.wait_ready(port, cfg["health_path"])
        version = lib.fetch_version(port, cfg["version_method"])
        (cell_dir / "version.json").write_text(json.dumps(version, indent=2))
        lib.warmup(port, WARMUP, ZERO_ROW)

        tdir = TARGETS_ROOT / "built" / f"{pipeline}_host"
        tdir.mkdir(parents=True, exist_ok=True)
        url = lib.vegeta_url_for(port)
        lib.build_targets(BODIES, url, tdir / "targets.json")

        report = lib.vegeta_attack(targets_dir=tdir, out_dir=cell_dir,
                                   rate=rps, duration_s=duration, cpuset="2,3")
        lat = report.get("latencies", {})
        summary = {
            "layer": "L2_host_http", "pipeline": pipeline, "variant": label,
            "workers": 1, "threads": 2, "rps_target": rps, "duration_s": duration,
            "requests": report.get("requests", 0),
            "rps_achieved": report.get("throughput", 0.0),
            "error_rate": round(1.0 - report.get("success", 0.0), 6),
            "status_codes": report.get("status_codes", {}),
            "p50_ms": lat.get("50th", 0) / 1e6,
            "p95_ms": lat.get("95th", 0) / 1e6,
            "p99_ms": lat.get("99th", 0) / 1e6,
            "mean_ms": lat.get("mean", 0) / 1e6,
            "model_sha256": version.get("model_sha256"),
        }
        (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[L2] {label}: P50={summary['p50_ms']:.2f} P99={summary['p99_ms']:.2f} "
              f"err={summary['error_rate']:.3%} ach={summary['rps_achieved']:.0f}rps")
        return summary
    finally:
        if os.name == "posix":
            try:
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        log.close()
        time.sleep(2)  # let the port free up before the next pipeline


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipelines", nargs="+", default=["fastapi", "bentoml"])
    p.add_argument("--max-latency-ms-list", nargs="+", type=int, default=[5],
                   help="BentoML variants by max_latency_ms (ignored for fastapi)")
    p.add_argument("--rps", type=int, default=150)
    p.add_argument("--duration", type=int, default=45)
    args = p.parse_args()

    if not BODIES.exists():
        print(f"[L2] missing {BODIES}; run targets/generate_targets.py first", file=sys.stderr)
        return 1

    for pipe in args.pipelines:
        if pipe == "fastapi":
            run_pipeline("fastapi", args.rps, args.duration)
        elif pipe == "bentoml":
            for lat in args.max_latency_ms_list:
                bjson = TARGETS_ROOT / "built" / f"bentoml_lat{lat}" / "batching.json"
                bjson.parent.mkdir(parents=True, exist_ok=True)
                bjson.write_text(json.dumps({
                    "max_batch_size": 16, "max_latency_ms": lat,
                    "_note": f"L2 host variant max_latency_ms={lat}",
                }, indent=2))
                run_pipeline("bentoml", args.rps, args.duration,
                             label=f"bentoml_lat{lat}", batching_path=bjson)
        else:
            raise ValueError(f"unknown pipeline: {pipe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
