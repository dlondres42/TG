"""Layer 3 (mechanism subset) — re-runs 20 hand-picked cells with docker-stats
CPU sampling enabled, writing to a separate results tree so the original L3
dataset is untouched.

Why this exists: the thesis claims "BentoML's single dispatcher coroutine per
worker serialises requests; CPU is *not* saturated at collapse". That claim is
mechanism-level and needs CPU-utilization evidence, not just latency. The
original L3 sweep captured only Vegeta-side data; this script re-runs a
mechanism-critical subset with server-side `docker stats` sampled at ~1 Hz.

Cell selection (20 cells, 3 repeats each, 45 s attack = ~1.5 h wall time):
  fastapi          (5)  reference at saturation:    w1t1@600, w1t2@600, w2t1@600, w2t2@600, w2t2@1000
  bentoml_lat50    (8)  collapse band, headline:    w1t1@350,600 / w1t2@350,600 / w2t1@600,1000 / w2t2@600,1000
  bentoml_lat250   (4)  looser deadline:            w1t2@350,600 / w2t2@600,1000
  bentoml_lat5     (3)  calibration artifact:       w1t2@150,350 / w2t2@600

Outputs land under results/layer3_with_cpu/<variant>/w<W>_t<T>/rps<RPS>/run<N>/
with the same artifacts as L3 (result.bin, summary.json, version.json) plus
cpu_pct.csv. The original results/layer3/ tree is left untouched.

Usage:
    uv run python layers/L3_cpu_mechanism.py
    uv run python layers/L3_cpu_mechanism.py --repeats 2 --duration 30   # quick smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402
from layers.L3_docker_sweep import (  # noqa: E402
    build_variants, summarize, PORTS, SERVER_CPUSET, VEGETA_CPUSET,
    WARMUP, ZERO_ROW, SETTLE_S,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "layer3_with_cpu"
TARGETS_ROOT = Path(__file__).resolve().parents[1] / "targets"
BODIES = TARGETS_ROOT / "bodies.jsonl"

# (variant_label, workers, threads, rps) — the 20 mechanism-critical cells.
CELLS: list[tuple[str, int, int, int]] = [
    # FastAPI reference at saturation (5)
    ("fastapi",        1, 1,  600),
    ("fastapi",        1, 2,  600),
    ("fastapi",        2, 1,  600),
    ("fastapi",        2, 2,  600),
    ("fastapi",        2, 2, 1000),
    # BentoML lat50 — collapse band, the headline variant (8)
    ("bentoml_lat50",  1, 1,  350),
    ("bentoml_lat50",  1, 1,  600),
    ("bentoml_lat50",  1, 2,  350),
    ("bentoml_lat50",  1, 2,  600),
    ("bentoml_lat50",  2, 1,  600),
    ("bentoml_lat50",  2, 1, 1000),
    ("bentoml_lat50",  2, 2,  600),
    ("bentoml_lat50",  2, 2, 1000),
    # BentoML lat250 — looser deadline, batching has room (4)
    ("bentoml_lat250", 1, 2,  350),
    ("bentoml_lat250", 1, 2,  600),
    ("bentoml_lat250", 2, 2,  600),
    ("bentoml_lat250", 2, 2, 1000),
    # BentoML lat5 — calibration-artifact reference (3)
    ("bentoml_lat5",   1, 2,  150),
    ("bentoml_lat5",   1, 2,  350),
    ("bentoml_lat5",   2, 2,  600),
]


def run_cell_with_cpu(variant: dict, workers: int, threads: int, rps: int,
                     duration: int, repeat: int, targets_dir: Path, force: bool) -> dict | None:
    label = variant["label"]
    pipeline = variant["pipeline"]
    cell_dir = RESULTS / label / f"w{workers}_t{threads}" / f"rps{rps}" / f"run{repeat}"
    if (cell_dir / "summary.json").exists() and (cell_dir / "cpu_pct.csv").exists() and not force:
        return json.loads((cell_dir / "summary.json").read_text())

    port = PORTS[pipeline]
    name = f"tg-{label}-cpu"
    cfg = lib.PIPELINES[pipeline]
    cell_dir.mkdir(parents=True, exist_ok=True)
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

        (cell_dir / "version.json").write_text(json.dumps(version, indent=2))
        # Sampler covers the entire vegeta attack window; idle samples before/after
        # are dominated by the warmup ramp and would noise the steady-state mean.
        # interval_s=0: docker stats --no-stream itself blocks ~1 s per sample,
        # so a 0 sleep gives the maximum native ~1 Hz density.
        with lib.DockerCpuSampler(name, cell_dir / "cpu_pct.csv", interval_s=0.0):
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
        cpu_stats = lib.cpu_summary_from_csv(cell_dir / "cpu_pct.csv")
        summary.update(cpu_stats)
        (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        cpu_mean = cpu_stats.get("cpu_mean_pct", float("nan"))
        cpu_max = cpu_stats.get("cpu_max_pct", float("nan"))
        print(
            f"[CPU] {label} w{workers}t{threads} rps{rps} run{repeat}: "
            f"P50={summary['p50_ms']:.2f} P99={summary['p99_ms']:.2f} "
            f"err={summary['error_rate']:.3%} "
            f"cpu_mean={cpu_mean:.1f}% peak={cpu_max:.1f}%"
        )
        return summary
    finally:
        lib.docker_rm(name)
        time.sleep(SETTLE_S)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=int, default=45)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--force", action="store_true")
    p.add_argument("--only-cells", type=int, default=None,
                   help="Limit to the first N cells (smoke testing)")
    args = p.parse_args()

    if not BODIES.exists():
        print(f"[CPU] missing {BODIES}; run targets/generate_targets.py first", file=sys.stderr)
        return 1

    # Build variants for every pipeline/max_latency we touch — the dispatcher
    # config is otherwise baked into per-variant batching.json files.
    lat_set = sorted({int(v.split("lat")[1]) for v, *_ in CELLS if v.startswith("bentoml_lat")})
    variants_by_label = {
        v["label"]: v for v in build_variants(["fastapi", "bentoml"], lat_set)
    }

    # Vegeta targets once per pipeline.
    targets_dirs = {}
    for pipe in ("fastapi", "bentoml"):
        tdir = TARGETS_ROOT / "built" / pipe
        tdir.mkdir(parents=True, exist_ok=True)
        url = lib.vegeta_url_for(PORTS[pipe])
        n = lib.build_targets(BODIES, url, tdir / "targets.json")
        targets_dirs[pipe] = tdir
        print(f"[CPU] built {n} targets for {pipe} -> {url}")

    cells = CELLS if args.only_cells is None else CELLS[: args.only_cells]
    print(f"[CPU] running {len(cells)} cells x {args.repeats} repeats x {args.duration}s "
          f"-> results/layer3_with_cpu/")

    t0 = time.monotonic()
    n_runs, n_fail = 0, 0
    for label, w, t, rps in cells:
        v = variants_by_label[label]
        for r in range(1, args.repeats + 1):
            try:
                run_cell_with_cpu(v, w, t, rps, args.duration, r,
                                  targets_dirs[v["pipeline"]], args.force)
                n_runs += 1
            except Exception as e:
                n_fail += 1
                print(f"[CPU] {label} w{w}t{t} rps{rps} run{r}: FAILED ({e!r})")
                lib.docker_rm(f"tg-{label}-cpu")
                time.sleep(SETTLE_S)
    print(f"[CPU] done: {n_runs} runs, {n_fail} failed, "
          f"{(time.monotonic() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
