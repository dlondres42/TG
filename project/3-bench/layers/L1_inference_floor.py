"""Layer 1 — pure ONNX inference floor (no HTTP, no framework).

Measures the model's intrinsic CPU cost by calling onnxruntime.InferenceSession
directly in a tight loop, using the same SessionOptions the servers use
(intra_op=2, inter_op=1, ORT_ENABLE_ALL, sequential). This is the physical
floor below which no serving stack can go on this hardware.

Also measures *batched* floors (B in {1,4,8,16}) so the BentoML latency budget
can subtract the inference cost matching the batch size the dispatcher actually
forms, rather than the single-row floor (see thesis/phase2.md decomposition
caveat). Reports both total per-call latency and per-row latency.

Output: results/layer1/{L1_summary.json, L1_histogram.png}

Usage:
    uv run python layers/L1_inference_floor.py
    uv run python layers/L1_inference_floor.py --iters 20000 --threads 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "layer1"

BATCH_SIZES = [1, 4, 8, 16]
PERCENTILES = [50, 95, 99, 99.9]


def build_session(threads: int) -> tuple[ort.InferenceSession, str]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(
        str(lib.ARTIFACTS / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"],
    )
    return sess, sess.get_inputs()[0].name


def sample_batch(b: int) -> np.ndarray:
    import pandas as pd
    df = pd.read_parquet(lib.TEST_PARQUET)
    cols = [c for c in df.columns if c != "label"]
    return df.sample(n=b, random_state=0)[cols].to_numpy(dtype=np.float32)


def measure(sess, input_name: str, arr: np.ndarray, iters: int) -> list[float]:
    feed = {input_name: arr}
    # warmup
    for _ in range(50):
        sess.run(None, feed)
    lat_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        sess.run(None, feed)
        lat_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return lat_ms


def pctls(xs: list[float]) -> dict:
    a = np.asarray(xs)
    return {f"p{p}": float(np.percentile(a, p)) for p in PERCENTILES} | {"mean": float(a.mean())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--threads", type=int, default=2, help="ORT intra_op_num_threads")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sess, input_name = build_session(args.threads)

    results = {}
    b1_latencies: list[float] = []
    for b in BATCH_SIZES:
        arr = sample_batch(b)
        lat = measure(sess, input_name, arr, args.iters)
        if b == 1:
            b1_latencies = lat
        per_call = pctls(lat)
        per_row = {k: (v / b if k != "mean" else v / b) for k, v in per_call.items()}
        results[str(b)] = {"per_call_ms": per_call, "per_row_ms": per_row}
        print(
            f"[L1] batch={b:>2}  per-call P50={per_call['p50']:.3f} "
            f"P99={per_call['p99']:.3f} P99.9={per_call['p99.9']:.3f} ms | "
            f"per-row P50={per_row['p50']:.4f} ms"
        )

    summary = {
        "iters": args.iters,
        "threads": args.threads,
        "onnxruntime": ort.__version__,
        "model_sha256": json.loads((lib.ARTIFACTS / "schema.json").read_text())["model_sha256"],
        "batch_sizes": BATCH_SIZES,
        "results": results,
    }
    (OUT_DIR / "L1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[L1] wrote {OUT_DIR / 'L1_summary.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.clip(b1_latencies, 0, np.percentile(b1_latencies, 99.5)), bins=80)
        ax.set_xlabel("single-row inference latency (ms)")
        ax.set_ylabel("count")
        ax.set_title(f"L1 inference floor (batch=1, {args.threads} ORT threads)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "L1_histogram.png", dpi=120)
        print(f"[L1] wrote {OUT_DIR / 'L1_histogram.png'}")
    except Exception as e:  # pragma: no cover
        print(f"[L1] histogram skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
