"""FastAPI + ONNX Runtime inference server.

Loads the Phase 1 ONNX artifact under a lifespan handler, verifies its SHA256
against schema.json (cross-phase contract guard), and exposes:

  GET  /healthz   liveness + model SHA
  GET  /version   server stack + matrix knobs as actually applied
  POST /predict   {"features": [[..28..], ...]} -> {"scores": [..]}

Threading is locked at process start from env:
  WEB_CONCURRENCY                uvicorn worker count (set in entrypoint.sh)
  ORT_INTRA_OP_NUM_THREADS       intra-op pool size
  ORT_INTER_OP_NUM_THREADS       inter-op pool size (1, locked per proposal)
  FASTAPI_THREADPOOL_SIZE        AnyIO sync-handler threadpool capacity

The handler is `def`, not `async def`. session.run releases the GIL but is
CPU-bound, so an async handler would serialise requests inside one event loop
and conflate the worker axis with an asyncio artifact.

The AnyIO sync-handler threadpool (Starlette dispatches sync `def` handlers
through `anyio.to_thread`) is a third concurrency knob between WEB_CONCURRENCY
and ORT_INTRA_OP_NUM_THREADS. Its default is 40 tokens per worker. We pin it
explicitly via FASTAPI_THREADPOOL_SIZE so the value is recorded per cell and
cannot drift between runs; at the RPS levels in Phase 3 the in-flight count
stays well below this cap so it is not the limiting factor.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from anyio import to_thread
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/model.onnx"))
SCHEMA_PATH = Path(os.environ.get("SCHEMA_PATH", "/app/schema.json"))
FASTAPI_THREADPOOL_SIZE = int(os.environ.get("FASTAPI_THREADPOOL_SIZE", "32"))

N_FEATURES = 28


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_session() -> tuple[ort.InferenceSession, str, str, dict]:
    schema = json.loads(SCHEMA_PATH.read_text())
    expected_sha = schema["model_sha256"]
    actual_sha = _sha256(MODEL_PATH)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"model SHA mismatch: schema={expected_sha} actual={actual_sha}"
        )

    intra = int(os.environ.get("ORT_INTRA_OP_NUM_THREADS", "2"))
    inter = int(os.environ.get("ORT_INTER_OP_NUM_THREADS", "1"))

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra
    opts.inter_op_num_threads = inter
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sess = ort.InferenceSession(
        str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name

    applied = {
        "intra_op_num_threads": intra,
        "inter_op_num_threads": inter,
        "web_concurrency": int(os.environ.get("WEB_CONCURRENCY", "1")),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "fastapi_threadpool_size": FASTAPI_THREADPOOL_SIZE,
    }
    return sess, input_name, actual_sha, applied


@asynccontextmanager
async def lifespan(app: FastAPI):
    to_thread.current_default_thread_limiter().total_tokens = FASTAPI_THREADPOOL_SIZE
    sess, input_name, sha, applied = _build_session()
    app.state.session = sess
    app.state.input_name = input_name
    app.state.model_sha = sha
    app.state.applied = applied
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[list[float]]


@app.get("/healthz")
def healthz(request: Request):
    return {"status": "ok", "model_sha256": request.app.state.model_sha}


@app.get("/version")
def version(request: Request):
    return {
        "stack": "fastapi",
        "onnxruntime": ort.__version__,
        "model_sha256": request.app.state.model_sha,
        "applied": request.app.state.applied,
    }


@app.post("/predict")
def predict(body: PredictRequest, request: Request):
    arr = np.asarray(body.features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"features must be 2-D with shape (N, {N_FEATURES}); got {arr.shape}",
        )
    sess: ort.InferenceSession = request.app.state.session
    input_name: str = request.app.state.input_name
    outputs = sess.run(None, {input_name: arr})
    probs = outputs[1][:, 1]
    return probs.tolist()


@app.exception_handler(HTTPException)
async def http_exc_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
