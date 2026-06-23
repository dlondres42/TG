"""BentoML + ONNX Runtime + adaptive batching.

Phase 2B. Loads the Phase 1 ONNX artifact, verifies SHA against schema.json,
and exposes a single batchable /predict endpoint.

Configurable via env (read at process start, frozen for the run):
  WEB_CONCURRENCY            -> @bentoml.service workers
  ORT_INTRA_OP_NUM_THREADS   -> ORT session intra-op pool
  ORT_INTER_OP_NUM_THREADS   -> ORT session inter-op pool

Adaptive batching parameters live in batching.json (calibrated once, then
held fixed across the matrix per the Phase 2 plan):
  {"max_batch_size": <int>, "max_latency_ms": <int>}
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import bentoml
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/model.onnx"))
SCHEMA_PATH = Path(os.environ.get("SCHEMA_PATH", "/app/schema.json"))
BATCHING_PATH = Path(os.environ.get("BATCHING_PATH", str(Path(__file__).with_name("batching.json"))))

N_FEATURES = 28


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_batching() -> dict:
    if BATCHING_PATH.exists():
        return json.loads(BATCHING_PATH.read_text())
    # Pre-calibration placeholder: tiny batch + tiny wait = effectively no batching.
    return {"max_batch_size": 1, "max_latency_ms": 1}


_BATCHING = _load_batching()
_WORKERS = int(os.environ.get("WEB_CONCURRENCY", "1"))


@bentoml.service(
    name="tg_serving_bentoml",
    workers=_WORKERS,
    traffic={"timeout": 30},
)
class TgService:
    def __init__(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text())
        expected = schema["model_sha256"]
        actual = _sha256(MODEL_PATH)
        if actual != expected:
            raise RuntimeError(
                f"model SHA mismatch: schema={expected} actual={actual}"
            )

        intra = int(os.environ.get("ORT_INTRA_OP_NUM_THREADS", "2"))
        inter = int(os.environ.get("ORT_INTER_OP_NUM_THREADS", "1"))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra
        opts.inter_op_num_threads = inter
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.sess = ort.InferenceSession(
            str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"],
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.model_sha = actual
        self.applied = {
            "intra_op_num_threads": intra,
            "inter_op_num_threads": inter,
            "web_concurrency": _WORKERS,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "batching": _BATCHING,
        }

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=_BATCHING["max_batch_size"],
        max_latency_ms=_BATCHING["max_latency_ms"],
    )
    def predict(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != N_FEATURES:
            raise ValueError(
                f"features must be 2-D (N, {N_FEATURES}); got shape {arr.shape}"
            )
        outputs = self.sess.run(None, {self.input_name: arr})
        return outputs[1][:, 1].astype(np.float32)

    @bentoml.api
    def healthz(self) -> dict:
        return {"status": "ok", "model_sha256": self.model_sha}

    @bentoml.api
    def version(self) -> dict:
        return {
            "stack": "bentoml",
            "bentoml": bentoml.__version__,
            "onnxruntime": ort.__version__,
            "model_sha256": self.model_sha,
            "applied": self.applied,
        }
