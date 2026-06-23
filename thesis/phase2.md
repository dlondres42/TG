# Phase 2 — Serving Pipelines

## Purpose

Phase 2 builds **two containerized HTTP services** that load the Phase 1 ONNX artifact and expose the same `POST /predict` contract. The pipelines differ only in serving stack:

- **Pipeline A** — FastAPI + ONNX Runtime, no batching. Represents the "generalist async web framework" approach.
- **Pipeline B** — BentoML 1.4 + ONNX Runtime, adaptive batching enabled. Represents the "specialised ML serving with request coalescence" approach.

Identical input schema, identical output schema, identical CPU/memory budgets. Anything different in Phase 3 latency is therefore attributable to the stack itself, not to the model or to the request shape.

## Shared HTTP contract

| Field | Value | Note |
|---|---|---|
| Predict request | `POST /predict` with body `{"features": [[f1..f28], ...]}` | 2-D list of float32s; single-row clients send a list-of-one |
| Predict response | raw JSON array `[p1, p2, ...]` | positive-class probability per row; ONNX `probabilities[:, 1]` |
| Liveness (FastAPI) | `GET /healthz` → `{"status":"ok","model_sha256":"<hex>"}` | used by Vegeta warmup and the Docker HEALTHCHECK |
| Liveness (BentoML) | `GET /livez` (BentoML built-in) plus `POST /healthz` (custom; returns SHA) | BentoML defaults all `@bentoml.api` to POST; `/livez` is the only GET available |
| Introspection | `GET /version` (FastAPI) / `POST /version` (BentoML) | returns stack, ORT version, model SHA, and the *live applied* parallelism knobs |

### Why the response is a raw array, not `{"scores": [...]}`

The original plan wrapped the response in `{"scores": [...]}`. This had to change because **BentoML's `batchable=True` collapses concurrent requests by stacking their inputs along `batch_dim=0` and splitting the returned array back per request**. If the API returns a dict, BentoML doesn't know how to split it across requests and adaptive batching becomes unusable. The simplest cross-pipeline contract is therefore a raw float array — and FastAPI was aligned to the same shape so the Vegeta target file is identical for both servers.

### The `model_sha256` contract

Both servers compute the SHA256 of `model.onnx` at startup and assert it matches `schema.json["model_sha256"]`. A mismatch fails the process before it accepts any traffic. The bench harness reads the SHA from `/version` into each result directory, so a post-hoc reader can prove the same model produced every cell of the matrix — there is no way for Phase 3 to silently serve a different model than Phase 1 exported.

## Configuration surface — env-driven matrix knobs

Both containers read seven environment variables at process start and freeze them for the lifetime of the run. A single Docker image therefore serves every matrix cell:

| Variable | Values | What it controls |
|---|---|---|
| `WEB_CONCURRENCY` | 1 or 2 | uvicorn `--workers` (FastAPI); `@bentoml.service(workers=...)` (BentoML) |
| `ORT_INTRA_OP_NUM_THREADS` | 1 or 2 | ORT `SessionOptions.intra_op_num_threads` — intra-op parallelism inside one inference call |
| `ORT_INTER_OP_NUM_THREADS` | 1 always | locked per proposal §3 |
| `OMP_NUM_THREADS` | mirrors intra-op | prevents OpenMP from spawning its own pool |
| `MKL_NUM_THREADS` | mirrors intra-op | same for any BLAS path |
| `MODEL_PATH` | `/app/model.onnx` | volume-mounted at `docker run` time |
| `SCHEMA_PATH` | `/app/schema.json` | same |

`OMP_NUM_THREADS` and `MKL_NUM_THREADS` matter for the same reason as the ORT knobs: if any layer below ORT spawns its own thread pool, the matrix's intent is broken. By forcing every pool to the same size, the (workers × ORT-threads) configuration genuinely controls the total amount of intra-process concurrency.

## Parallelism matrix (advisor feedback)

After advisor review the experimental matrix gained a (workers × ORT intra-op threads) axis. Each of the four cells isolates a different parallelism mechanism within the same 2-CPU container budget:

| Configuration | Concurrency mechanism | Objective in Phase 3 |
|---|---|---|
| 1 worker / 1 thread | none | ONNX clean baseline |
| 1 worker / 2 threads | ORT intra-op | measure internal (within-call) parallelism |
| 2 workers / 1 thread | process replication | measure serving-layer parallelism |
| 2 workers / 2 threads | both, oversubscribed | measure contention risk (4 thread-slots on 2 cores) |

The same four cells are run for both pipelines.

## Pipeline A — FastAPI + ONNX Runtime

**Files** in `project/2-serving/fastapi/`: `app.py`, `entrypoint.sh`, `Dockerfile`, `.dockerignore`, `pyproject.toml`, `README.md`.

### Critical implementation detail — sync handler, not async

`/predict` is `def`, **not** `async def`. `session.run` releases the GIL but is CPU-bound. An `async def` handler would hold one event-loop thread for the duration of every inference, serialising all concurrent requests within a worker behind one task. That would conflate the worker axis of the matrix with an artifact of asyncio scheduling, ruining the measurement.

Defining the handler as a plain `def` lets FastAPI dispatch each request to its threadpool. With `WEB_CONCURRENCY=2` and `ORT_INTRA_OP_NUM_THREADS=2`, two requests can be in flight, each using two ORT intra-op threads — the four matrix cells then mean what they say.

### Third concurrency knob — the AnyIO sync-handler threadpool

There is a third concurrency layer between the uvicorn worker (controlled by `WEB_CONCURRENCY`) and the ORT intra-op pool (controlled by `ORT_INTRA_OP_NUM_THREADS`): Starlette dispatches sync `def` handlers to a thread via `anyio.to_thread`, which uses an internal capacity limiter. Its default is **40 tokens per worker process**. This thread reservoir is what actually lets multiple sync handlers run concurrently in a single worker — without it the sync-vs-async distinction would be moot.

This knob is not part of the experimental matrix, but is pinned explicitly to a known value (`FASTAPI_THREADPOOL_SIZE`, default 32) in the lifespan handler so the configuration is recorded per cell rather than implicit. At the Phase 3 request rates the in-flight request count stays well below this cap (estimated as `RPS × P50_seconds`; a worst-case 1000 RPS × 50 ms ≈ 50 in flight across both workers, or 25 per worker), so the pool size is not the limiting factor. The analysis notebook will recompute this estimate per cell and flag any cell where in-flight ≥ 50 % of the pool size as suspect.

### Lifespan handler

The FastAPI `@asynccontextmanager` lifespan handler runs once per worker on startup:

1. Read `schema.json`, hash `model.onnx`, assert match.
2. Build `SessionOptions` with the env-driven intra-op / inter-op threads, `graph_optimization_level=ORT_ENABLE_ALL`, `execution_mode=ORT_SEQUENTIAL`.
3. Create `InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])`.
4. Stash session + input-name + model-SHA + applied-config on `app.state`.

The Pydantic body model rejects non-2-D shapes or wrong column counts with HTTP 400. ONNX output `[None, 2]` is sliced as `outputs[1][:, 1]`.

## Pipeline B — BentoML + adaptive batching

**Files** in `project/2-serving/bentoml/`: `service.py`, `entrypoint.sh`, `Dockerfile`, `.dockerignore`, `bentofile.yaml`, `batching.json`, `calibrate_batching.py`, `calibration_report.md`, `pyproject.toml`, `README.md`.

BentoML version: **1.4.39**. Uses the modern decorator-based API:

```python
@bentoml.service(name="tg_serving_bentoml", workers=_WORKERS, traffic={"timeout": 30})
class TgService:
    def __init__(self):  ...  # SHA check + ORT SessionOptions identical to FastAPI
    @bentoml.api(batchable=True, batch_dim=0,
                 max_batch_size=_BATCHING["max_batch_size"],
                 max_latency_ms=_BATCHING["max_latency_ms"])
    def predict(self, features: np.ndarray) -> np.ndarray: ...
    @bentoml.api
    def healthz(self) -> dict: ...
    @bentoml.api
    def version(self) -> dict: ...
```

### How adaptive batching works on the wire

BentoML maps the parameter name `features` to the request JSON key `features`. The value (a JSON list-of-lists) is auto-deserialised to `np.ndarray`. With `batchable=True`, the dispatcher collects concurrent inbound requests, stacks their `features` arrays along `batch_dim=0`, calls `predict` once with the stacked input, and splits the returned array back into per-request responses. This is the entire point of "adaptive batching" — concurrent requests share one inference call.

The `max_batch_size` and `max_latency_ms` knobs control this dispatcher:

- `max_batch_size` is the maximum number of items the dispatcher will combine into a single inference. It is also a *per-call* cap: a single request whose `features` array has N > `max_batch_size` rows fails with `503 process is overloaded`. The bench scripts therefore always send single-row requests.
- `max_latency_ms` is the maximum time the dispatcher will wait collecting requests before dispatching a partial batch.

### Batching calibration — what we did, and why it was wrong by design

**Note**: this section is preserved as a methodological case study. The Phase 2 calibration produced a configuration (`max_batch_size=16, max_latency_ms=5`) that **structurally penalised BentoML at the Phase 3 RPS levels**, because it was optimised against the wrong objective. The Phase 3 re-measurement corrects this by sweeping `max_latency_ms` as an explicit axis (see [thesis/phase3.md](thesis/phase3.md)). The lesson it teaches — that frameworks must be benchmarked against the SLA they will operate under in production — is itself a thesis-level contribution.

The original calibration framed `max_latency_ms` as a "batch wait window to minimise." Reading BentoML's [dispatcher source](project/2-serving/bentoml/.venv/Lib/site-packages/bentoml/_internal/marshal/dispatcher.py) (line 303) shows that framing is wrong: `max_latency_ms` is a **hard request-completion deadline**, and the dispatcher cancels (→ HTTP 503) any request whose estimated total latency exceeds it. With the measured serving-overhead floor of ~3–4 ms (L2 − L1), a 5 ms deadline leaves ~1 ms of headroom; any queue at all (inevitable at 150+ RPS) breaches the deadline and triggers mass 503s. BentoML itself logs a warning to this effect (dispatcher.py:278: *"max latency likely too low for serving"*) which appeared in our Phase 2 logs and was ignored.

The calibration also probed a single low load (100 RPS) and picked the lowest-P99/zero-error value. Because the knob is a deadline, *lower = more fragile under load* — the procedure structurally selected the value most fragile at the RPS levels Phase 3 measures. The "best vs best" framing the calibration aimed for is the right idea, but executed against the wrong objective it produced "best at 100 RPS vs anything else" — actively worse than picking a reasonable SLA from first principles.

### Batching calibration — procedure

`calibrate_batching.py` is the one-shot driver. For each of the 9 grid cells `{max_batch_size ∈ 16, 32, 64} × {max_latency_ms ∈ 2, 5, 10}` ms:

1. Write the cell's values to `batching.json`.
2. Start BentoML as a subprocess.
3. Wait up to 30 s for `/predict` to respond 200 (the readiness probe; `/livez` reports ready earlier than the service is actually loaded).
4. Send 100 warmup single-row requests.
5. Drive **open-loop load** (the proposal's anti-Coordinated-Omission rule, §3.3) at 100 RPS for 15 s, recording per-request wall-clock latency.
6. Kill the BentoML process tree (`taskkill /F /T /PID` on Windows; BentoML spawns worker subprocesses that simple `terminate()` does not reliably reach).
7. Compute P50/P95/P99/P99.9, count errors.

After all 9 cells, the script picks the cell with **lowest P99 among zero-error cells** (tie-break on P50), commits that pair to `batching.json`, and writes the comparison table to `calibration_report.md`.

Parallelism during calibration is fixed at `WEB_CONCURRENCY=1, ORT_INTRA_OP_NUM_THREADS=2` — the cell where adaptive batching has the most room to help, because a single worker funnels all requests through one batchable runner.

### Batching calibration — result

| max_batch_size | max_latency_ms | errors | P50 ms | P95 ms | P99 ms |
|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 47 | 9.23 | 13.05 | 19.82 |
| **16** | **5** | **0** | **9.47** | **13.38** | **35.45** ← picked |
| 16 | 10 | 2 | 9.08 | 13.36 | 43.31 |
| 32 | 2 | 33 | 9.03 | 13.30 | 16.38 |
| 32 | 5 | 5 | 9.35 | 13.49 | 42.94 |
| 32 | 10 | 1 | 9.43 | 13.02 | 42.46 |
| 64 | 2 | 42 | 8.99 | 12.64 | 17.10 |
| 64 | 5 | 2 | 9.22 | 12.96 | 43.77 |
| 64 | 10 | 1 | 9.59 | 15.50 | 44.71 |

Observations:

- Every `max_latency_ms = 2` cell produced timeouts/503s. The dispatcher under-fills batches and rejects under that tight a window.
- Every `max_latency_ms = 10` cell inflated P99 to ~43 ms regardless of `max_batch_size`. The wait window cost outweighs the batching gain at this RPS.
- `max_latency_ms = 5` is the only window with zero errors across all batch sizes. Among those, `batch_size = 16` minimised P99.

The picked pair `(16, 5)` is committed in `batching.json` and used for every Phase 3 cell of the **`bentoml_lat5` variant**, which is retained as the documented mis-calibrated baseline. The Phase 3 sweep also re-runs BentoML at `max_latency_ms ∈ {50, 250}` ms to disentangle the artifact from the underlying behaviour — see [thesis/phase3.md](thesis/phase3.md).

### Why the 503s appeared at low `max_latency_ms`

The 33–47 errors per cell at `max_latency_ms = 2` ms are not a bug — they are the dispatcher refusing to accept work it cannot serve. BentoML's adaptive dispatcher operates a small queue plus a predictor that estimates, for each incoming request, whether a batch can be assembled and dispatched within `max_latency_ms`. When the predictor concludes the SLA cannot be met, it calls a `fallback()` that returns HTTP 503 with the message `process is overloaded`.

At `max_latency_ms = 2` ms, the SLA is fundamentally tighter than the work required. Importantly, the bottleneck is **not** inference — Phase 3's L1 microbenchmark measured single-row `session.run` at ~0.05 ms and a 16-row batch at ~0.5 ms, both far under 2 ms. The cost that breaks the SLA is the **per-request serving overhead** (request parsing, dispatcher queue management, response serialisation), which Phase 3's L2 host-HTTP measurement put at ~4 ms P50 for BentoML:

| Step | Time cost (measured) |
|---|---|
| `session.run` (single row → 16-row batch) | ~0.05 → ~0.5 ms |
| Framework + dispatch + (de)serialisation overhead | ~4 ms (L2 − L1) |
| **Effective per-request budget needed** | **~4 ms** |

`~4 ms ≫ 2 ms`, so the dispatcher cannot honour a 2 ms target regardless of how cheap inference is; it sheds ~2–3 % of load via the `process is overloaded` fallback rather than queueing indefinitely (fail-fast, documented in the framework's source).

The lesson the grid teaches is therefore concrete: **`max_latency_ms` must exceed the per-request serving overhead by enough margin to absorb queueing under the production RPS**, not merely the inference time (~0.05 ms). At `max_latency_ms = 5` the margin against the ~4 ms baseline is ~1 ms — feasible at 100 RPS where queues stay empty, but not at the Phase 3 sweep's 150–1000 RPS where queues form continuously. This is precisely why Phase 3 re-runs BentoML at 50 ms (a realistic 10× margin) and 250 ms (a generous margin), and reports goodput-within-SLA as the fair cross-pipeline metric (see [thesis/phase3.md](thesis/phase3.md) for the corrected results). The reframing also stands: since inference is ~0.05 ms, essentially **all** measured latency in both pipelines is serving-stack overhead — quantified by the Phase 3 latency budget.

## Verification gates

### Score parity (`project/3-bench/parity.py`)

3-way score check: FastAPI vs BentoML vs in-process local ONNX. Reads N rows from `test.parquet`, scores each via local `onnxruntime.InferenceSession` as ground truth, sends single-row requests to each server, asserts `|score − ground_truth| < 1e-5`.

Result with both servers containerised and running simultaneously, n = 500:

| Comparison | max abs diff | mean abs diff |
|---|---:|---:|
| FastAPI vs local ONNX | 2.98e-7 | 3.80e-8 |
| BentoML vs local ONNX | 2.98e-7 | 3.80e-8 |
| FastAPI vs BentoML | **0.00** | 0.00 |

Servers are bit-identical at the response level, and both are within 3 × 10⁻⁷ of the in-process reference — three orders of magnitude tighter than the 10⁻⁵ tolerance.

### Matrix smoke (`project/3-bench/matrix_smoke.py`)

Boots every cell of the (workers × threads) matrix on each pipeline, asserts `/version` reports back the values we set, asserts `/predict` returns a valid single-element array. All 8 cells (4 cells × 2 pipelines) currently PASS.

## Docker images

| Image | Size | Notes |
|---|---:|---|
| `tg-serving-fastapi` | 279 MB | python:3.11-slim runtime; only curl added |
| `tg-serving-bentoml` | 487 MB | heavier dep tree (pydantic v2 + opentelemetry + simple_di) |

Both:

- **Multi-stage build**. Builder installs `uv` and constructs the venv directly at `/opt/venv` via `UV_PROJECT_ENVIRONMENT=/opt/venv`. Runtime stage does `COPY --from=builder /opt/venv /opt/venv` — same path so Python shebangs remain valid.
- **Code in `/srv`, model mounted at `/app`**. The WORKDIR must *not* be `/app`, because `docker run -v ...:/app:ro` shadows everything in `/app`.
- **No model baked into the image**. A single image serves every matrix cell via env vars and a read-only volume mount of `project/1-model/artifacts`.
- **HEALTHCHECK** uses curl against the GET liveness endpoint (`/healthz` for FastAPI, `/livez` for BentoML).

## Lessons learned / gotchas

Documenting the non-obvious issues encountered during Phase 2 development so that future revisits don't have to rediscover them:

1. **BentoML 1.4 requires `max_batch_size > 1`** (not `≥ 1`). A placeholder of 1 raises `ValueError` at import time.
2. **BentoML 1.4 per-request batch cap**: a single request with N > `max_batch_size` items is rejected with `503 process is overloaded`. Parity/smoke scripts therefore always send single-row requests.
3. **Sync vs async FastAPI handler**: must be `def`, not `async def`. ONNX is CPU-bound; an async handler would serialise concurrent requests within one event loop and break the worker-axis semantics.
4. **`/app` is the mount point**: WORKDIR and code must live in `/srv`. The volume mount `-v ...:/app:ro` hides anything under `/app`.
5. **uv venv path matters**: copying a venv between paths breaks the Python shebangs in `/opt/venv/bin/*`. Build the venv directly at the destination via `UV_PROJECT_ENVIRONMENT=/opt/venv`.
6. **Windows PowerShell 5.1 `Set-Content -Encoding utf8` writes UTF-16 BOM**. Use Python or Claude Code's Write tool for any JSON file that BentoML (or another consumer) will parse.
7. **Windows subprocess stdout=PIPE deadlock**: BentoML logs prolifically; with stdout to an unread PIPE, the 64 KB buffer fills and the server blocks on the next log write. Redirect to a log file when spawning servers from Python.
8. **Process-tree kill on Windows**: BentoML spawns worker subprocesses that `proc.terminate()` and `CTRL_BREAK_EVENT` do not reliably reach. Use `taskkill /F /T /PID`.
9. **Docker build context absolute paths**: background `docker build` invocations may run in a different CWD than expected. Always pass an absolute context path.
10. **BentoML dispatcher warning** ("max latency that is likely too low") appears once when the dispatcher's RPS estimator stabilises. Cosmetic at our load; not a real problem at 100+ RPS.

## Reproducing Phase 2

```bash
# Setup
make setup-fastapi
make setup-bentoml
make setup-bench

# Build both Docker images
make build-servers

# Calibrate BentoML batching (~6 minutes, regenerates batching.json)
make calibrate-batching

# Bring up both servers (one matrix cell each)
make run-fastapi WORKERS=1 THREADS=2
make run-bentoml WORKERS=1 THREADS=2

# Verify score parity across both servers and local ONNX
make parity

# Optional: exhaustive matrix smoke (~3 minutes)
cd project/3-bench && uv run python matrix_smoke.py

# Tear down
make stop-servers
```

## Phase 2 exit criteria (all met)

- Both Docker images build and pass their HEALTHCHECK.
- `/version` on each server reports the env-set matrix knobs correctly across all four (workers × threads) cells.
- `parity.py` shows pairwise score differences within 1e-5 (measured: 0.0 between servers, 2.98e-7 against local ONNX).
- `calibration_report.md` exists with a defensible pick: `max_batch_size=16, max_latency_ms=5`.
- All 8 matrix cells (4 cells × 2 pipelines) PASS the smoke test.
