"""Layer 0 — noise floor. Pure HTTP loopback against /healthz, no inference.

Quantifies the absolute measurement floor — TCP + HTTP + Vegeta scheduling +
container port-mapping — so that L1/L2/L3 numbers can be reported with an
honest "anything below ~X ms is at the noise floor" caveat in the thesis.

Recipe: a single FastAPI server at (w=1, t=1) on the same Docker path L3 uses,
hit by Vegeta at 1 RPS for 60 s against /healthz (cheap 200 OK with no model
inference). The pipeline choice is arbitrary — /healthz does no model work, so
FastAPI vs BentoML floors only differ by the framework's stat-200 path, which
is itself a finding worth recording.

Usage:
    uv run python layers/L0_noise_floor.py
    uv run python layers/L0_noise_floor.py --pipelines fastapi bentoml --duration 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "layer0"
TARGETS_ROOT = Path(__file__).resolve().parents[1] / "targets" / "built" / "noise_floor"
PORT = 8020


def write_healthz_targets(url: str, out_path: Path) -> None:
    """Vegeta JSON-format targets file with a single GET /healthz (or POST for BentoML).

    Vegeta cycles through targets line by line; one line is enough since the
    same target fires at 1 RPS for the whole window.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = {"method": "GET", "url": url, "header": {"Accept": ["application/json"]}}
    # Empty body for GET; vegeta accepts no body field for GET targets.
    out_path.write_text(json.dumps(target) + "\n")


def run_floor(pipeline: str, duration_s: int) -> dict:
    cfg = lib.PIPELINES[pipeline]
    out_dir = RESULTS / pipeline
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"noise_floor_{pipeline}"

    lib.run_server(pipeline, name=name, host_port=PORT, cpuset="0,1",
                   workers=1, threads=1)
    try:
        lib.wait_ready(PORT, cfg["health_path"])
        # /healthz path differs between pipelines (FastAPI: GET /healthz;
        # BentoML: GET /livez); use whichever the lib already exposes.
        url = f"http://{'127.0.0.1' if lib._NATIVE_VEGETA else 'host.docker.internal'}:{PORT}{cfg['health_path']}"
        write_healthz_targets(url, TARGETS_ROOT / pipeline / "targets.json")
        report = lib.vegeta_attack(
            targets_dir=TARGETS_ROOT / pipeline,
            out_dir=out_dir, rate=1, duration_s=duration_s, cpuset="2,3",
        )
        lat = report.get("latencies", {})
        summary = {
            "layer": "L0_noise_floor", "pipeline": pipeline,
            "endpoint": cfg["health_path"], "rps_target": 1, "duration_s": duration_s,
            "requests": report.get("requests", 0),
            "rps_achieved": report.get("throughput", 0.0),
            "error_rate": round(1.0 - report.get("success", 0.0), 6),
            "status_codes": report.get("status_codes", {}),
            "p50_ms": lat.get("50th", 0) / 1e6,
            "p95_ms": lat.get("95th", 0) / 1e6,
            "p99_ms": lat.get("99th", 0) / 1e6,
            "mean_ms": lat.get("mean", 0) / 1e6,
            "max_ms": lat.get("max", 0) / 1e6,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[L0] {pipeline}: P50={summary['p50_ms']:.3f} P99={summary['p99_ms']:.3f} "
              f"max={summary['max_ms']:.3f} err={summary['error_rate']:.3%}")
        return summary
    finally:
        lib.docker_rm(name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipelines", nargs="+", default=["fastapi", "bentoml"])
    p.add_argument("--duration", type=int, default=60)
    args = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rollup = {}
    for pipe in args.pipelines:
        rollup[pipe] = run_floor(pipe, args.duration)
    (RESULTS / "L0_summary.json").write_text(json.dumps(rollup, indent=2))
    print(f"[L0] wrote {RESULTS / 'L0_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
