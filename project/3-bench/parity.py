"""Parity check across (server A, server B, local ONNX ground truth).

Phase 2 acceptance gate. Reads N rows from the Phase 1 test parquet, scores them
against the local ONNX model (ground truth) and against any servers passed on
the command line, and asserts all scores agree within tolerance.

Usage:
    uv run python parity.py                              # FastAPI only
    uv run python parity.py --bentoml http://...:8002    # FastAPI + BentoML
    uv run python parity.py --fastapi http://...:8001 --bentoml http://...:8002

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "1-model" / "artifacts"
TEST_PARQUET = ROOT / "1-model" / "data" / "test.parquet"

TOLERANCE = 1e-5
DEFAULT_N = 1000
BATCH = 1  # single-row requests; matches Phase 3 traffic and BentoML's per-request batch cap


def load_rows(n: int) -> tuple[np.ndarray, list[list[float]]]:
    df = pd.read_parquet(TEST_PARQUET)
    feature_cols = [c for c in df.columns if c != "label"]
    sample = df.sample(n=n, random_state=0)
    arr = sample[feature_cols].to_numpy(dtype=np.float32)
    return arr, arr.tolist()


def local_scores(arr: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(
        str(ARTIFACTS / "model.onnx"), providers=["CPUExecutionProvider"],
    )
    outputs = sess.run(None, {"features": arr})
    return outputs[1][:, 1].astype(np.float32)


def server_scores(client: httpx.Client, url: str, rows: list[list[float]]) -> np.ndarray:
    out: list[float] = []
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        r = client.post(f"{url}/predict", json={"features": chunk}, timeout=30.0)
        r.raise_for_status()
        body = r.json()
        scores = body["scores"] if isinstance(body, dict) else body
        out.extend(scores)
    return np.asarray(out, dtype=np.float32)


def check_healthz(client: httpx.Client, url: str, *, method: str = "GET") -> str:
    if method == "POST":
        r = client.post(f"{url}/healthz", json={}, timeout=5.0)
    else:
        r = client.get(f"{url}/healthz", timeout=5.0)
    r.raise_for_status()
    return r.json()["model_sha256"]


def compare(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> bool:
    diff = np.abs(a - b)
    mx = float(diff.max())
    mn = float(diff.mean())
    ok = mx < TOLERANCE
    flag = "OK" if ok else "FAIL"
    print(f"[{flag}] {name_a} vs {name_b}: max={mx:.3e} mean={mn:.3e} (tol {TOLERANCE:.0e})")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fastapi", default="http://127.0.0.1:8001", help="FastAPI server URL")
    p.add_argument("--bentoml", default=None, help="BentoML server URL (optional)")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="rows to check")
    p.add_argument("--skip-fastapi", action="store_true")
    args = p.parse_args()

    schema = json.loads((ARTIFACTS / "schema.json").read_text())
    expected_sha = schema["model_sha256"]

    arr, rows = load_rows(args.n)
    print(f"[parity] loaded {len(rows)} rows from {TEST_PARQUET.name}")

    truth = local_scores(arr)
    print(f"[parity] local ONNX scores computed (mean={truth.mean():.4f})")

    targets: list[tuple[str, str]] = []
    if not args.skip_fastapi:
        targets.append(("fastapi", args.fastapi))
    if args.bentoml:
        targets.append(("bentoml", args.bentoml))

    if not targets:
        print("[parity] no servers selected", file=sys.stderr)
        return 2

    results: dict[str, np.ndarray] = {}
    all_ok = True
    with httpx.Client() as client:
        for name, url in targets:
            method = "POST" if name == "bentoml" else "GET"
            sha = check_healthz(client, url, method=method)
            if sha != expected_sha:
                print(f"[FAIL] {name} /healthz SHA mismatch: {sha} != {expected_sha}")
                all_ok = False
                continue
            print(f"[parity] {name} /healthz SHA OK ({sha[:16]}...)")
            scores = server_scores(client, url, rows)
            results[name] = scores
            if not compare(name, scores, "local-onnx", truth):
                all_ok = False

    if "fastapi" in results and "bentoml" in results:
        if not compare("fastapi", results["fastapi"], "bentoml", results["bentoml"]):
            all_ok = False

    print()
    print("[parity] PASS" if all_ok else "[parity] FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
