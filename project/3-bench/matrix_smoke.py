"""Boot every (workers x ORT-thread) matrix cell of each pipeline once.

For each (pipeline, workers, threads) cell:
  1. docker rm any leftover container with the same name
  2. docker run -d with the env vars driving the cell
  3. wait for liveness (FastAPI: GET /healthz; BentoML: GET /livez)
  4. GET /version (FastAPI) / POST /version (BentoML), assert the live applied
     web_concurrency / intra_op_num_threads match what we set
  5. POST /predict with a single zero row, assert HTTP 200 and a 1-element response
  6. docker rm -f
  7. report the cell as PASS or FAIL

This is the Phase 2 exit gate. Phase 3's bench harness will reuse the same
docker-run invocation pattern (with CPU pinning) when measuring latency.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "1-model" / "artifacts"

MATRIX: list[tuple[int, int]] = [(1, 1), (1, 2), (2, 1), (2, 2)]

PIPELINES = {
    "fastapi": {
        "image": "tg-serving-fastapi",
        "container": "tg-fa-smoke",
        "port": 8101,
        "cpuset": "0,1",
        "health_method": "GET",
        "health_path": "/healthz",
        "version_method": "GET",
    },
    "bentoml": {
        "image": "tg-serving-bentoml",
        "container": "tg-bm-smoke",
        "port": 8102,
        "cpuset": "2,3",
        "health_method": "GET",
        "health_path": "/livez",
        "version_method": "POST",
    },
}

READY_TIMEOUT_S = 60
ZERO_ROW = [0.0] * 28


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=check)


def docker_rm(container: str) -> None:
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True)


def docker_run(pipeline: dict, workers: int, threads: int) -> None:
    docker_rm(pipeline["container"])
    args = [
        "docker", "run", "--rm", "-d",
        "--name", pipeline["container"],
        "--cpus=2", "--memory=2g", f"--cpuset-cpus={pipeline['cpuset']}",
        "-v", f"{ARTIFACTS}:/app:ro",
        "-e", f"WEB_CONCURRENCY={workers}",
        "-e", f"ORT_INTRA_OP_NUM_THREADS={threads}",
        "-e", "ORT_INTER_OP_NUM_THREADS=1",
        "-e", f"OMP_NUM_THREADS={threads}",
        "-e", f"MKL_NUM_THREADS={threads}",
        "-p", f"{pipeline['port']}:8000",
        pipeline["image"],
    ]
    run_cmd(args)


def wait_ready(client: httpx.Client, pipeline: dict) -> None:
    url = f"http://127.0.0.1:{pipeline['port']}{pipeline['health_path']}"
    deadline = time.monotonic() + READY_TIMEOUT_S
    last_err = ""
    while time.monotonic() < deadline:
        try:
            r = client.get(url, timeout=2.0)
            if r.status_code == 200:
                return
            last_err = f"status={r.status_code}"
        except httpx.HTTPError as e:
            last_err = str(e)
        time.sleep(1.0)
    raise TimeoutError(f"{pipeline['container']} not ready in {READY_TIMEOUT_S}s ({last_err})")


def fetch_version(client: httpx.Client, pipeline: dict) -> dict:
    url = f"http://127.0.0.1:{pipeline['port']}/version"
    if pipeline["version_method"] == "POST":
        r = client.post(url, json={}, timeout=5.0)
    else:
        r = client.get(url, timeout=5.0)
    r.raise_for_status()
    return r.json()


def call_predict(client: httpx.Client, pipeline: dict) -> list:
    url = f"http://127.0.0.1:{pipeline['port']}/predict"
    r = client.post(url, json={"features": [ZERO_ROW]}, timeout=10.0)
    r.raise_for_status()
    body = r.json()
    return body["scores"] if isinstance(body, dict) else body


def smoke_cell(name: str, pipeline: dict, workers: int, threads: int) -> tuple[bool, str]:
    cell = f"{name} w={workers} t={threads}"
    print(f"[smoke] {cell}: starting container")
    try:
        docker_run(pipeline, workers, threads)
    except subprocess.CalledProcessError as e:
        return False, f"docker run failed: {e.stderr.strip()}"

    try:
        with httpx.Client() as client:
            wait_ready(client, pipeline)
            version = fetch_version(client, pipeline)
            applied = version.get("applied", {})
            got_w = applied.get("web_concurrency")
            got_t = applied.get("intra_op_num_threads")
            if got_w != workers:
                return False, f"web_concurrency mismatch: env={workers} applied={got_w}"
            if got_t != threads:
                return False, f"intra_op_num_threads mismatch: env={threads} applied={got_t}"
            if name == "fastapi":
                got_tp = applied.get("fastapi_threadpool_size")
                if not isinstance(got_tp, int) or got_tp <= 0:
                    return False, f"fastapi_threadpool_size missing/invalid: {got_tp!r}"

            scores = call_predict(client, pipeline)
            if not (isinstance(scores, list) and len(scores) == 1 and isinstance(scores[0], (int, float))):
                return False, f"predict returned bad shape: {scores!r}"

        return True, f"applied={applied}"
    except Exception as e:
        return False, repr(e)
    finally:
        docker_rm(pipeline["container"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=list(PIPELINES.keys()), default=None,
                   help="restrict to one pipeline")
    args = p.parse_args()

    pipelines = {args.only: PIPELINES[args.only]} if args.only else PIPELINES
    results: list[tuple[str, int, int, bool, str]] = []
    all_ok = True
    for name, pipeline in pipelines.items():
        for workers, threads in MATRIX:
            ok, detail = smoke_cell(name, pipeline, workers, threads)
            results.append((name, workers, threads, ok, detail))
            print(f"[smoke]   {'PASS' if ok else 'FAIL'}: {detail}")
            if not ok:
                all_ok = False

    print()
    print("Matrix smoke summary:")
    print(f"  {'pipeline':<10} {'workers':>7} {'threads':>7}  result")
    print(f"  {'-' * 35}")
    for name, w, t, ok, _ in results:
        print(f"  {name:<10} {w:>7} {t:>7}  {'PASS' if ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
