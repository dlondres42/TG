"""Saturation pilot — find each cell's knee RPS to calibrate the L3 sweep.

For each (pipeline, workers, threads) cell, ramp the Vegeta rate through a
ladder until the cell saturates (error rate > 1% or P99 > P99_KNEE_MS). The
last rate before saturation is the usable ceiling; the recommended L3 sweep is
5 rates evenly spaced from 10% to 90% of that ceiling (plan §3.8).

Short per-step duration (this is a calibration probe, not a measurement).

Output: results/pilot/saturation.json + a printed table.

Usage:
    uv run python layers/saturation_pilot.py
    uv run python layers/saturation_pilot.py --duration 10 --cells 1x2 2x2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "pilot"
TARGETS_ROOT = Path(__file__).resolve().parents[1] / "targets"
BODIES = TARGETS_ROOT / "bodies.jsonl"

MATRIX = [(1, 1), (1, 2), (2, 1), (2, 2)]
PORTS = {"fastapi": 8021, "bentoml": 8022}
SERVER_CPUSET = "0,1"
VEGETA_CPUSET = "2,3"
LADDER = [100, 250, 500, 1000, 1500, 2000, 3000, 4000]
ERR_KNEE = 0.01      # >1% errors = saturated
P99_KNEE_MS = 200.0  # >200ms P99 on a sub-ms model = saturated
WARMUP = 100
ZERO_ROW = [0.0] * 28


def suggested_sweep(ceiling: int) -> list[int]:
    if ceiling <= 0:
        return []
    return [int(round(x / 10) * 10) for x in np.linspace(0.1 * ceiling, 0.9 * ceiling, 5)]


def ramp_cell(pipeline, workers, threads, duration, targets_dir) -> dict:
    port = PORTS[pipeline]
    cfg = lib.PIPELINES[pipeline]
    name = f"tg-{pipeline}-pilot"
    lib.run_server(pipeline, name=name, host_port=port, cpuset=SERVER_CPUSET,
                   workers=workers, threads=threads)
    ladder_results = []
    ceiling = 0
    knee = None
    try:
        lib.wait_ready(port, cfg["health_path"])
        lib.warmup(port, WARMUP, ZERO_ROW)
        for rps in LADDER:
            out = RESULTS / pipeline / f"w{workers}_t{threads}" / f"rps{rps}"
            report = lib.vegeta_attack(targets_dir=targets_dir, out_dir=out,
                                       rate=rps, duration_s=duration, cpuset=VEGETA_CPUSET)
            lat = report.get("latencies", {})
            err = round(1.0 - report.get("success", 0.0), 6)
            p99 = lat.get("99th", 0) / 1e6
            ladder_results.append({"rps": rps, "error_rate": err, "p99_ms": round(p99, 3),
                                   "rps_achieved": round(report.get("throughput", 0.0), 1)})
            saturated = err > ERR_KNEE or p99 > P99_KNEE_MS
            print(f"[pilot] {pipeline} w{workers}t{threads} rps{rps}: "
                  f"P99={p99:.1f}ms err={err:.2%} {'<< KNEE' if saturated else ''}")
            if saturated:
                knee = rps
                break
            ceiling = rps
    finally:
        lib.docker_rm(name)

    return {
        "ceiling_rps": ceiling,          # last clean rate
        "knee_rps": knee,                # first saturated rate (None if never)
        "ladder": ladder_results,
        "suggested_sweep": suggested_sweep(ceiling),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipelines", nargs="+", default=["fastapi", "bentoml"])
    p.add_argument("--duration", type=int, default=8, help="seconds per ramp step")
    p.add_argument("--cells", nargs="+", default=None, help="subset like 1x2 2x2")
    args = p.parse_args()

    if not BODIES.exists():
        print(f"[pilot] missing {BODIES}; run targets/generate_targets.py first", file=sys.stderr)
        return 1

    cells = MATRIX
    if args.cells:
        want = {tuple(int(x) for x in c.split("x")) for c in args.cells}
        cells = [c for c in MATRIX if c in want]

    targets_dirs = {}
    for pipe in args.pipelines:
        tdir = TARGETS_ROOT / "built" / f"{pipe}_pilot"
        tdir.mkdir(parents=True, exist_ok=True)
        lib.build_targets(BODIES, f"http://host.docker.internal:{PORTS[pipe]}/predict", tdir / "targets.json")
        targets_dirs[pipe] = tdir

    out = {}
    for pipe in args.pipelines:
        for (w, t) in cells:
            out[f"{pipe}/w{w}_t{t}"] = ramp_cell(pipe, w, t, args.duration, targets_dirs[pipe])

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "saturation.json").write_text(json.dumps(out, indent=2))

    print("\n=== Saturation summary ===")
    print(f"  {'cell':<22} {'ceiling':>8} {'knee':>6}   suggested sweep")
    for k, v in out.items():
        knee = v["knee_rps"] if v["knee_rps"] is not None else ">ladder"
        print(f"  {k:<22} {v['ceiling_rps']:>8} {str(knee):>6}   {v['suggested_sweep']}")
    print(f"\n[pilot] wrote {RESULTS / 'saturation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
