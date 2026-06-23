"""Export trained XGBoost model to ONNX and verify XGBoost<->ONNX prediction parity.

The exported artifact is the contract between Phase 1 and Phase 2 — both servers
load this exact .onnx file (matched by SHA256 in schema.json) so any latency
difference between FastAPI and BentoML is attributable to serving stack, not model.

Outputs:
  artifacts/model.onnx     — ONNX model, opset 15, float32 [None, 28] input
  artifacts/schema.json    — input/output spec + SHA256 + parity report
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxmltools
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

N_FEATURES = 28
TARGET_OPSET = 15
PARITY_SAMPLE_SIZE = 10_000
PARITY_TOLERANCE = 1e-5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(booster_path: Path, onnx_path: Path) -> None:
    print(f"[export] loading {booster_path}")
    model = xgb.XGBClassifier()
    model.load_model(str(booster_path))

    initial_types = [("features", FloatTensorType([None, N_FEATURES]))]
    onnx_model = onnxmltools.convert_xgboost(
        model, initial_types=initial_types, target_opset=TARGET_OPSET,
    )
    onnxmltools.utils.save_model(onnx_model, str(onnx_path))
    print(f"[export] wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")


def parity_check(booster_path: Path, onnx_path: Path) -> dict:
    model = xgb.XGBClassifier()
    model.load_model(str(booster_path))

    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    feature_cols = [c for c in test_df.columns if c != "label"]
    n = min(PARITY_SAMPLE_SIZE, len(test_df))
    sample = test_df.sample(n=n, random_state=0)
    X = sample[feature_cols].to_numpy(dtype=np.float32)

    py_proba = model.predict_proba(X)[:, 1].astype(np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_outputs = sess.run(None, {"features": X})
    onnx_proba = _extract_positive_proba(onnx_outputs).astype(np.float32)

    diff = np.abs(py_proba - onnx_proba)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"[parity] sample={n}  max_abs_diff={max_diff:.3e}  mean_abs_diff={mean_diff:.3e}")
    if max_diff >= PARITY_TOLERANCE:
        raise AssertionError(
            f"parity check FAILED: max_abs_diff={max_diff:.3e} >= {PARITY_TOLERANCE:.3e}"
        )
    print(f"[parity] OK (tolerance {PARITY_TOLERANCE:.0e})")

    return {
        "sample_size": int(n),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "tolerance": PARITY_TOLERANCE,
    }


def _extract_positive_proba(outputs: list) -> np.ndarray:
    proba_out = outputs[1]
    if not (isinstance(proba_out, np.ndarray) and proba_out.ndim == 2):
        raise RuntimeError(f"unexpected ONNX probability output: {type(proba_out)}")
    return proba_out[:, 1]


def write_schema(onnx_path: Path, schema_path: Path, parity: dict) -> None:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    schema = {
        "input_name": "features",
        "input_dtype": "float32",
        "input_shape": [None, N_FEATURES],
        "output_names": [o.name for o in sess.get_outputs()],
        "n_features": N_FEATURES,
        "target_opset": TARGET_OPSET,
        "model_sha256": sha256(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "parity": parity,
    }
    schema_path.write_text(json.dumps(schema, indent=2))
    print(f"[schema] wrote {schema_path} (sha256={schema['model_sha256'][:16]}...)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify-only", action="store_true",
                   help="Skip ONNX conversion; just re-run the parity check on existing model.onnx.")
    args = p.parse_args()

    booster_path = ARTIFACTS / "model.ubj"
    onnx_path = ARTIFACTS / "model.onnx"
    schema_path = ARTIFACTS / "schema.json"

    if not args.verify_only:
        convert(booster_path, onnx_path)
    parity = parity_check(booster_path, onnx_path)
    write_schema(onnx_path, schema_path, parity)


if __name__ == "__main__":
    main()
