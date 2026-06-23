# 5. Implementation

## 5.1 Phase 1: model and ONNX artifact

<<SOURCE: thesis/phase1.md — copy and polish; the hyperparameter table, the
dataset description, and the parity check are the high-value content.>>

### 5.1.1 The artifact contract

<<TODO: 1 paragraph stating the contract — `model.onnx` (opset 15, input
"features" `[None, 28]` float32) + `schema.json` (input/output names,
SHA-256 of the ONNX file, XGBoost-vs-ONNX parity report). Both servers
verify the SHA at startup. Mismatch = process fails to start.>>

### 5.1.2 Three packaging gotchas

<<SOURCE: thesis/phase1.md §"ONNX export gotchas" — copy verbatim; the
three onnxmltools / skl2onnx / xgboost incompatibilities are exactly the
kind of methodological detail an examiner will probe.>>

## 5.2 Pipeline A: FastAPI + ONNX Runtime

<<SOURCE: thesis/phase2.md §"Pipeline A".>>

### 5.2.1 The critical implementation detail: sync handler

The `/predict` handler is **`def`, not `async def`**. `session.run` releases
the GIL during native inference but is CPU-bound; an async handler would
pin all requests to the single event loop and serialise them. A sync
handler is dispatched by FastAPI to the AnyIO threadpool, letting uvicorn
workers and ORT intra-op threads compose as the matrix intends.

### 5.2.2 Lifespan

<<TODO: 1 paragraph on the lifespan handler — SHA verification, ORT
`SessionOptions` (intra_op from env, inter_op=1, ORT_ENABLE_ALL,
SEQUENTIAL), `InferenceSession` with `CPUExecutionProvider`, the session
+ input name + model SHA cached on `app.state`.>>

### 5.2.3 The wire contract

`POST /predict` body `{"features": [[f1..f28], ...]}`; response a raw JSON
array `[p1, p2, ...]`. The response shape was chosen for BentoML's
adaptive-batching split semantics (dictionary wrappers prevent the per-
request slice) and FastAPI was aligned to match so a single Vegeta target
serves both servers.

## 5.3 Pipeline B: BentoML + adaptive batching

<<SOURCE: thesis/phase2.md §"Pipeline B".>>

### 5.3.1 Service definition

```python
@bentoml.service(name="tg_serving_bentoml", workers=_WORKERS,
                 traffic={"timeout": 30})
class TgService:
    def __init__(self): ...   # same SHA + ORT session
    @bentoml.api(batchable=True, batch_dim=0,
                 max_batch_size=_BATCHING["max_batch_size"],
                 max_latency_ms=_BATCHING["max_latency_ms"])
    def predict(self, features: np.ndarray) -> np.ndarray: ...
```

<<TODO: 1 paragraph on what `batchable=True, batch_dim=0` does — collect
concurrent inbound requests, stack `features` arrays along axis 0, call
`predict` once with the stacked input, split the returned array per
request.>>

### 5.3.2 The dispatcher coroutine

BentoML's `_internal/marshal/dispatcher.py` defines `controller()` — a
single async coroutine per worker. It drains the request queue, decides
when to dispatch a batch (based on queue depth, time waited, and the
estimated cost from a running `train_optimizer` phase), then returns
results to the per-request futures. Every dispatch decision is serialised
through this one coroutine; this is the bottleneck identified in Chapter 6.

<<CITE: bentoml_dispatcher>> — source file at
`src/bentoml/_internal/marshal/dispatcher.py`.

### 5.3.3 `max_latency_ms`: hard deadline, not wait window

The relevant lines (`dispatcher.py:303-311`):

```python
latency_0 = w0 + a * n + b
if n > 1 and latency_0 >= self.max_latency:
    self._queue.popleft().future.cancel()  # → fallback() → 503
```

A queued request whose *estimated total wait + service time* would exceed
`max_latency_ms` is cancelled and returned as HTTP 503. This is a hard
deadline; the original calibration mis-interpreted it as a wait-window-to-
minimise and selected a 5 ms value that systematically sheds load. The
Phase 3 correction promoted the knob to an explicit axis.

## 5.4 The bench harness

<<SOURCE: project/3-bench/lib.py, layers/L0..L3 scripts, analyze.py.>>

### 5.4.1 Module structure

```
project/3-bench/
  lib.py                    server lifecycle, Vegeta runner, docker stats sampler
  stats.py                  bootstrap CIs, MWU, Cliff's δ, Little's Law, CCDF
  analyze.py                aggregate L1/L2/L3 results into summary.csv + figures
  cpu_mechanism_report.py   post-bench analysis + report generator
  layers/
    L0_noise_floor.py       1-RPS probe against /healthz, /livez (60 s)
    L1_inference_floor.py   tight Python loop, 10 000 iterations
    L2_host_http.py         host-pinned server + Vegeta (one cell)
    L3_docker_sweep.py      the 219-cell matrix sweep
    L3_cpu_mechanism.py     20-cell subset with docker stats sampling
  targets/                  Vegeta target body files (10 000 JSON bodies)
  results/                  layer0/, layer1/, layer2/, layer3/, layer3_with_cpu/
```

### 5.4.2 Per-cell artifacts

Each L3 cell produces:

- `result.bin` — raw Vegeta binary with every per-request latency and status
- `report.json` — Vegeta's text-report aggregates
- `summary.json` — parsed metrics, including the live `/version` payload
  with the applied parallelism knobs
- `version.json` — the server's introspection response (SHA, applied env)

For CPU-instrumented cells: `cpu_pct.csv` — the `docker stats` timeseries.

### 5.4.3 Resumability

The L3 sweep is idempotent: a cell with an existing `summary.json` is
skipped unless `--force` is passed. This was load-bearing when the
overnight CPU-mechanism re-run snapshot copy + parallel write pattern was
adopted — re-runs can be aborted and restarted without losing prior cells.

### 5.4.4 Containerisation

Both servers use the same multi-stage Dockerfile pattern: a builder stage
installs `uv` and builds the venv directly at `/opt/venv` (env
`UV_PROJECT_ENVIRONMENT=/opt/venv` — keeps shebang paths intact in the
runtime image); the runtime stage `COPY --from=builder /opt/venv /opt/venv`
and adds the application code to `/srv`. The model is mounted at `/app`
read-only at run time; `WORKDIR` is not `/app` because the volume mount
would otherwise shadow whatever the build placed there. The container CPU
budget is `--cpus=2 --memory=2g`; CPU pinning via `--cpuset-cpus=0,1`.

## 5.5 Reproducibility contract

<<TODO: 1 paragraph: the SHA-pinned model + per-cell `version.json` +
locked `uv.lock` per phase + Dockerfiles in-repo + Vegeta version pinned
+ the published harness scripts are sufficient to re-run any cell. State
this as the reproducibility claim; the appendix has the host specs.>>
