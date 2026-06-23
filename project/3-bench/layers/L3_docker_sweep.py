"""Layer 3 — containerised HTTP matrix sweep (the headline Phase 3 measurement).

Sweeps the (workers x ORT-threads) parallelism cells across an RPS grid for
each *pipeline variant*. FastAPI has a single variant; BentoML has one variant
per `max_latency_ms` value being tested (default {50, 250} ms — the 5 ms case
was the original calibration artifact and is preserved separately under
results/layer3/bentoml_lat5/).

For each (variant, workers, threads, rps, repeat):
  - start the serving container pinned to cores 0-1 (2 CPU / 2 GiB)
    (BentoML variants mount a per-variant batching.json + set BATCHING_PATH)
  - wait for liveness, capture /version, warm up
  - run a Vegeta sidecar pinned to cores 2-3 (open-loop, anti-Coordinated-Omission)
  - persist result.bin (raw), report.json (Vegeta), version.json, summary.json
  - tear the container down

Resumable: a cell with an existing summary.json is skipped unless --force.
Early-abort: once mean error rate > 50% at an RPS, higher RPS are skipped.

Usage:
    uv run python layers/L3_docker_sweep.py
    uv run python layers/L3_docker_sweep.py --pipelines bentoml --max-latency-ms-list 50 250
    uv run python layers/L3_docker_sweep.py --pipelines bentoml --max-latency-ms-list 50 \
        --cells 1x2 --rps 150 --duration 10 --repeats 1   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "layer3"
TARGETS_ROOT = Path(__file__).resolve().parents[1] / "targets"
BODIES = TARGETS_ROOT / "bodies.jsonl"

MATRIX = [(1, 1), (1, 2), (2, 1), (2, 2)]
PORTS = {"fastapi": 8001, "bentoml": 8002}
SERVER_CPUSET = "0,1"
VEGETA_CPUSET = "2,3"
WARMUP = 100  # bumped from 30 to fully drain BentoML's dispatcher train_optimizer phase
ZERO_ROW = [0.0] * 28
COLLAPSE_ERR = 0.5
SETTLE_S = 3
DEFAULT_MAX_BATCH_SIZE = 16


def ns_to_ms(x) -> float:
    return float(x) / 1e6


def build_variants(pipelines: list[str], max_latency_ms_list: list[int]) -> list[dict]:
    """Expand --pipelines into concrete variants.

    fastapi → one variant ("fastapi"); bentoml → one per max_latency_ms value.
    Each BentoML variant writes a per-variant batching.json that the server
    container mounts via -v ... :/work/batching.json:ro and reads via
    BATCHING_PATH=/work/batching.json.
    """
    variants: list[dict] = []
    for pipe in pipelines:
        if pipe == "fastapi":
            variants.append({"pipeline": "fastapi", "label": "fastapi",
                             "extra_mounts": None, "extra_env": None})
        elif pipe == "bentoml":
            for lat in max_latency_ms_list:
                bjson = TARGETS_ROOT / "built" / f"bentoml_lat{lat}" / "batching.json"
                bjson.parent.mkdir(parents=True, exist_ok=True)
                bjson.write_text(json.dumps({
                    "max_batch_size": DEFAULT_MAX_BATCH_SIZE,
                    "max_latency_ms": lat,
                    "_note": f"Phase 3 variant max_latency_ms={lat}",
                }, indent=2))
                variants.append({
                    "pipeline": "bentoml",
                    "label": f"bentoml_lat{lat}",
                    "extra_mounts": [(str(bjson), "/work/batching.json")],
                    "extra_env": {"BATCHING_PATH": "/work/batching.json"},
                })
        else:
            raise ValueError(f"unknown pipeline: {pipe}")
    return variants


def summarize(report: dict, meta: dict, version: dict) -> dict:
    lat = report.get("latencies", {})
    return {
        **meta,
        "requests": report.get("requests", 0),
        "rps_achieved": report.get("throughput", 0.0),
        "success_ratio": report.get("success", 0.0),
        "error_rate": round(1.0 - report.get("success", 0.0), 6),
        "status_codes": report.get("status_codes", {}),
        "p50_ms": ns_to_ms(lat.get("50th", 0)),
        "p90_ms": ns_to_ms(lat.get("90th", 0)),
        "p95_ms": ns_to_ms(lat.get("95th", 0)),
        "p99_ms": ns_to_ms(lat.get("99th", 0)),
        "max_ms": ns_to_ms(lat.get("max", 0)),
        "mean_ms": ns_to_ms(lat.get("mean", 0)),
        "model_sha256": version.get("model_sha256"),
        "applied": version.get("applied"),
    }


def run_cell(variant, workers, threads, rps, duration, repeat, targets_dir, force):
    label = variant["label"]
    pipeline = variant["pipeline"]
    cell_dir = RESULTS / label / f"w{workers}_t{threads}" / f"rps{rps}" / f"run{repeat}"
    if (cell_dir / "summary.json").exists() and not force:
        return json.loads((cell_dir / "summary.json").read_text())

    port = PORTS[pipeline]
    name = f"tg-{label}-l3"
    cfg = lib.PIPELINES[pipeline]
    lib.run_server(
        pipeline, name=name, host_port=port, cpuset=SERVER_CPUSET,
        workers=workers, threads=threads,
        extra_mounts=variant.get("extra_mounts"),
        extra_env=variant.get("extra_env"),
    )
    try:
        lib.wait_ready(port, cfg["health_path"])
        version = lib.fetch_version(port, cfg["version_method"])
        lib.warmup(port, WARMUP, ZERO_ROW)

        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "version.json").write_text(json.dumps(version, indent=2))
        report = lib.vegeta_attack(
            targets_dir=targets_dir, out_dir=cell_dir,
            rate=rps, duration_s=duration, cpuset=VEGETA_CPUSET,
        )
        meta = {
            "variant": label, "pipeline": pipeline,
            "workers": workers, "threads": threads,
            "rps_target": rps, "duration_s": duration, "repeat": repeat,
        }
        summary = summarize(report, meta, version)
        (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"[L3] {label} w{workers}t{threads} rps{rps} run{repeat}: "
            f"P50={summary['p50_ms']:.2f} P99={summary['p99_ms']:.2f} "
            f"err={summary['error_rate']:.3%} ach={summary['rps_achieved']:.0f}rps"
        )
        return summary
    finally:
        lib.docker_rm(name)
        time.sleep(SETTLE_S)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipelines", nargs="+", default=["fastapi", "bentoml"])
    p.add_argument("--max-latency-ms-list", nargs="+", type=int, default=[50, 250],
                   help="BentoML variants by max_latency_ms (ignored for fastapi)")
    p.add_argument("--rps", nargs="+", type=int, default=[50, 150, 350, 600, 1000])
    p.add_argument("--duration", type=int, default=45)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cells", nargs="+", default=None, help="subset of cells like 1x2 2x1")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if not BODIES.exists():
        print(f"[L3] missing {BODIES}; run targets/generate_targets.py first", file=sys.stderr)
        return 1

    cells = MATRIX
    if args.cells:
        want = {tuple(int(x) for x in c.split("x")) for c in args.cells}
        cells = [c for c in MATRIX if c in want]

    variants = build_variants(args.pipelines, args.max_latency_ms_list)

    # Build Vegeta targets once per *pipeline*. URL host depends on whether
    # Vegeta runs natively (localhost) or as a Docker sidecar (host.docker.internal).
    targets_dirs = {}
    for pipe in {v["pipeline"] for v in variants}:
        tdir = TARGETS_ROOT / "built" / pipe
        tdir.mkdir(parents=True, exist_ok=True)
        url = lib.vegeta_url_for(PORTS[pipe])
        n = lib.build_targets(BODIES, url, tdir / "targets.json")
        targets_dirs[pipe] = tdir
        print(f"[L3] built {n} targets for {pipe} -> {url}")

    print(f"[L3] variants to run: {[v['label'] for v in variants]}")

    t0 = time.monotonic()
    rps_sorted = sorted(args.rps)
    n_runs = 0
    n_fail = 0
    for v in variants:
        for (w, t) in cells:
            collapsed = False
            for rps in rps_sorted:
                if collapsed:
                    print(f"[L3] {v['label']} w{w}t{t} rps{rps}: skipped (cell saturated)")
                    continue
                errs = []
                for r in range(1, args.repeats + 1):
                    try:
                        summary = run_cell(v, w, t, rps, args.duration, r,
                                           targets_dirs[v["pipeline"]], args.force)
                        n_runs += 1
                        if isinstance(summary, dict) and summary.get("error_rate") is not None:
                            errs.append(summary["error_rate"])
                    except Exception as e:
                        n_fail += 1
                        print(f"[L3] {v['label']} w{w}t{t} rps{rps} run{r}: FAILED ({e!r})")
                        lib.docker_rm(f"tg-{v['label']}-l3")
                        time.sleep(SETTLE_S)
                if errs and (sum(errs) / len(errs)) > COLLAPSE_ERR:
                    collapsed = True
                    print(f"[L3] {v['label']} w{w}t{t}: collapse at rps{rps} "
                          f"(mean err {sum(errs)/len(errs):.1%}) -> skipping higher RPS")
    print(f"[L3] done: {n_runs} runs, {n_fail} failed, "
          f"{(time.monotonic() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
