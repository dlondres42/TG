"""Shared Phase 3 bench helpers: server lifecycle + Vegeta runner.

Used by matrix_smoke.py (Phase 2 gate) and the L2/L3 latency harness. All
server containers run with the proposal's fixed budget (2 CPU, 2 GiB) and
CPU pinning. Vegeta runs natively on the host with `taskset` pinning when
available (preferred — eliminates a layer of Docker indirection for the load
generator) and falls back to a Docker sidecar container otherwise. Both modes
keep the load generator on cores disjoint from the server.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]          # project/
ARTIFACTS = ROOT / "1-model" / "artifacts"
TEST_PARQUET = ROOT / "1-model" / "data" / "test.parquet"

VEGETA_IMAGE = "peterevans/vegeta:latest"

# Prefer native vegeta + taskset (Linux) over the Docker sidecar (Windows).
# `BENCH_FORCE_DOCKER_VEGETA=1` forces the sidecar even when native is available.
_NATIVE_VEGETA = (
    os.name == "posix"
    and shutil.which("vegeta") is not None
    and shutil.which("taskset") is not None
    and os.environ.get("BENCH_FORCE_DOCKER_VEGETA", "") != "1"
)

PIPELINES: dict[str, dict] = {
    "fastapi": {
        "image": "tg-serving-fastapi",
        "health_path": "/healthz",   # GET
        "version_method": "GET",
    },
    "bentoml": {
        "image": "tg-serving-bentoml",
        "health_path": "/livez",     # GET (BentoML built-in)
        "version_method": "POST",    # custom @bentoml.api endpoints are POST
    },
}

READY_TIMEOUT_S = 90


# --------------------------------------------------------------------------- #
# Server container lifecycle
# --------------------------------------------------------------------------- #
def docker_rm(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)


def run_server(
    pipeline: str, *, name: str, host_port: int, cpuset: str,
    workers: int, threads: int,
    extra_mounts: list[tuple[str, str]] | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Start a serving container detached. Mounts the Phase 1 artifacts read-only.

    extra_mounts: list of (host_path, container_path) to mount read-only — used
    by the L3 variant sweep to inject a per-variant batching.json.
    extra_env: additional env vars (e.g. BATCHING_PATH pointing at the mounted
    variant config).
    """
    cfg = PIPELINES[pipeline]
    docker_rm(name)
    args = [
        "docker", "run", "--rm", "-d",
        "--name", name,
        "--cpus=2", "--memory=2g", f"--cpuset-cpus={cpuset}",
        "-v", f"{ARTIFACTS}:/app:ro",
        "-e", f"WEB_CONCURRENCY={workers}",
        "-e", f"ORT_INTRA_OP_NUM_THREADS={threads}",
        "-e", "ORT_INTER_OP_NUM_THREADS=1",
        "-e", f"OMP_NUM_THREADS={threads}",
        "-e", f"MKL_NUM_THREADS={threads}",
    ]
    for src, dst in extra_mounts or []:
        args += ["-v", f"{src}:{dst}:ro"]
    for k, v in (extra_env or {}).items():
        args += ["-e", f"{k}={v}"]
    args += ["-p", f"{host_port}:8000", cfg["image"]]
    subprocess.run(args, capture_output=True, text=True, check=True)


def wait_ready(host_port: int, health_path: str, timeout: float = READY_TIMEOUT_S) -> None:
    url = f"http://127.0.0.1:{host_port}{health_path}"
    deadline = time.monotonic() + timeout
    last = ""
    with httpx.Client() as c:
        while time.monotonic() < deadline:
            try:
                if c.get(url, timeout=2.0).status_code == 200:
                    return
            except httpx.HTTPError as e:
                last = str(e)
            time.sleep(1.0)
    raise TimeoutError(f"server on :{host_port}{health_path} not ready in {timeout}s ({last})")


def fetch_version(host_port: int, version_method: str) -> dict:
    url = f"http://127.0.0.1:{host_port}/version"
    with httpx.Client() as c:
        r = c.post(url, json={}, timeout=5.0) if version_method == "POST" else c.get(url, timeout=5.0)
    r.raise_for_status()
    return r.json()


def warmup(host_port: int, n: int, row: list[float]) -> None:
    """Best-effort warmup; tolerates transient connection errors so a flaky
    warmup never crashes the sweep (one shared keep-alive client to minimise
    connection churn)."""
    url = f"http://127.0.0.1:{host_port}/predict"
    with httpx.Client() as c:
        for _ in range(n):
            try:
                c.post(url, json={"features": [row]}, timeout=10.0)
            except httpx.HTTPError:
                pass


# --------------------------------------------------------------------------- #
# Vegeta targets + attack
# --------------------------------------------------------------------------- #
def build_targets(bodies_path: Path, url: str, out_path: Path) -> int:
    """Convert a bodies.jsonl file into a Vegeta JSON-format targets file.

    Each output line is a Vegeta target with the request body base64-encoded
    inline, pointing at `url`. Returns the number of targets written.
    """
    n = 0
    with open(bodies_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            body_b64 = base64.b64encode(line.encode("utf-8")).decode("ascii")
            target = {
                "method": "POST",
                "url": url,
                "body": body_b64,
                "header": {"Content-Type": ["application/json"]},
            }
            fout.write(json.dumps(target) + "\n")
            n += 1
    return n


def vegeta_url_for(host_port: int) -> str:
    """Server URL Vegeta should hit. Native Vegeta reaches localhost directly;
    a Vegeta inside Docker must go through host.docker.internal."""
    host = "127.0.0.1" if _NATIVE_VEGETA else "host.docker.internal"
    return f"http://{host}:{host_port}/predict"


def vegeta_attack(
    *, targets_dir: Path, out_dir: Path, rate: int, duration_s: int, cpuset: str = "2,3",
) -> dict:
    """Attack the server and write result.bin + report.json into out_dir.

    Uses native Vegeta + taskset on Linux (preferred); falls back to a Docker
    sidecar on Windows. `-timeout=3s` caps each request so a saturated server
    fails fast instead of holding connections (which exhausts Windows
    ephemeral ports under the Docker-sidecar path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "result.bin"
    rep_path = out_dir / "report.json"

    if _NATIVE_VEGETA:
        # Native: taskset pins to the disjoint cpuset; vegeta reads/writes the
        # host filesystem directly. No Docker layer for the load generator.
        attack_cmd = (
            f"taskset -c {cpuset} vegeta attack -format=json "
            f"-targets={targets_dir / 'targets.json'} "
            f"-rate={rate} -duration={duration_s}s -timeout=3s -output={bin_path}"
        )
        report_cmd = f"vegeta report -type=json {bin_path}"
        subprocess.run(attack_cmd, shell=True, check=True)
        rep_out = subprocess.run(report_cmd, shell=True, capture_output=True, text=True, check=True).stdout
        rep_path.write_text(rep_out)
    else:
        # Docker sidecar (Windows fallback).
        inner = (
            "vegeta attack -format=json -targets=/targets/targets.json "
            f"-rate={rate} -duration={duration_s}s -timeout=3s -output=/work/result.bin && "
            "vegeta report -type=json /work/result.bin > /work/report.json"
        )
        args = [
            "docker", "run", "--rm",
            "--add-host=host.docker.internal:host-gateway",
            f"--cpuset-cpus={cpuset}", "--cpus=2",
            "-v", f"{targets_dir}:/targets:ro",
            "-v", f"{out_dir}:/work",
            "--entrypoint", "sh", VEGETA_IMAGE, "-c", inner,
        ]
        subprocess.run(args, capture_output=True, text=True, check=True)
    return json.loads(rep_path.read_text())


# --------------------------------------------------------------------------- #
# Docker stats sampler (server-side CPU + memory during the attack window)
# --------------------------------------------------------------------------- #
class DockerCpuSampler:
    """Context manager that polls `docker stats` against a running container
    and writes a per-cell cpu_pct.csv (columns: t_s, cpu_pct, mem_mb).

    `docker stats --no-stream` blocks ~1 s waiting for one sample, so the
    effective polling rate is ~1 Hz regardless of the requested interval —
    that's fine for a 45 s attack window where we care about mean and peak,
    not sub-second jitter. Sample failures are swallowed silently so the
    sampler can't crash the surrounding cell run.
    """

    def __init__(self, container_name: str, out_csv: Path, interval_s: float = 1.0):
        self.name = container_name
        self.out = Path(out_csv)
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows: list[list] = []

    def __enter__(self) -> "DockerCpuSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "cpu_pct", "mem_mb"])
            w.writerows(self._rows)

    def _run(self) -> None:
        t0 = time.monotonic()
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["docker", "stats", "--no-stream",
                     "--format", "{{.CPUPerc}}|{{.MemUsage}}", self.name],
                    capture_output=True, text=True, timeout=3,
                )
                line = r.stdout.strip()
                if line and "|" in line:
                    cpu_str, mem_str = line.split("|", 1)
                    cpu = float(cpu_str.rstrip("%").strip())
                    mem_used = mem_str.split("/")[0].strip()
                    mem_mb = self._parse_mem_mb(mem_used)
                    self._rows.append([round(time.monotonic() - t0, 2), cpu, mem_mb])
            except (subprocess.SubprocessError, ValueError, OSError):
                pass
            self._stop.wait(self.interval)

    @staticmethod
    def _parse_mem_mb(s: str) -> float:
        s = s.strip()
        try:
            if s.endswith("GiB"):
                return float(s[:-3]) * 1024
            if s.endswith("MiB"):
                return float(s[:-3])
            if s.endswith("KiB"):
                return float(s[:-3]) / 1024
            if s.endswith("B"):
                return float(s[:-1]) / (1024 * 1024)
        except ValueError:
            pass
        return float("nan")


def cpu_summary_from_csv(csv_path: Path) -> dict:
    """Read a cpu_pct.csv and return {mean, p50, p95, max, n_samples}.

    Used by analyze.py to fold per-cell CPU into summary.json without rereading
    the raw CSV everywhere.
    """
    import numpy as np
    if not Path(csv_path).exists():
        return {}
    vals = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                continue
            try:
                vals.append(float(row[1]))
            except (ValueError, IndexError):
                pass
    if not vals:
        return {}
    arr = np.array(vals, dtype=np.float64)
    return {
        "cpu_mean_pct": round(float(arr.mean()), 2),
        "cpu_p50_pct": round(float(np.percentile(arr, 50)), 2),
        "cpu_p95_pct": round(float(np.percentile(arr, 95)), 2),
        "cpu_max_pct": round(float(arr.max()), 2),
        "cpu_n_samples": int(arr.size),
    }


def vegeta_latencies_ns(bin_dir: Path, cpuset: str = "2,3") -> list[int]:
    """Encode an existing result.bin (in bin_dir) to per-request latencies (ns).

    Used by analysis to compute arbitrary percentiles (e.g. P99.9) and
    histograms that Vegeta's text/json report does not expose directly.
    """
    return [lat for lat, _code in vegeta_latency_status_ns(bin_dir, cpuset=cpuset)]


def vegeta_latency_status_ns(bin_dir: Path, cpuset: str = "2,3") -> list[tuple[int, int]]:
    """Encode result.bin (in bin_dir) to per-request (latency_ns, http_code).

    Used by analysis to compute goodput-under-SLA. Native Vegeta on Linux when
    available; Docker sidecar otherwise.
    """
    if _NATIVE_VEGETA:
        cmd = f"vegeta encode --to csv {bin_dir / 'result.bin'}"
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout
    else:
        args = [
            "docker", "run", "--rm",
            f"--cpuset-cpus={cpuset}",
            "-v", f"{bin_dir}:/work:ro",
            "--entrypoint", "sh", VEGETA_IMAGE, "-c",
            "vegeta encode --to csv /work/result.bin",
        ]
        out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    rows: list[tuple[int, int]] = []
    # vegeta encode csv columns: timestamp_ns, http_code, latency_ns, bytes_out, bytes_in, error, ...
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                rows.append((int(parts[2]), int(parts[1])))
            except ValueError:
                pass
    return rows
