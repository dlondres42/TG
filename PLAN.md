# High-Level Execution Plan — TG: P99 Tail Latency for Tabular ML Serving

## Context

The proposal in [pre_projeto/pre_projeto.tex](pre_projeto/pre_projeto.tex) commits to a quantitative comparison of **P99 tail latency** between two serving pipelines for an XGBoost model on tabular data: **FastAPI + ONNX Runtime** vs **BentoML + adaptive batching**. The deliverable is the empirical analysis of latency distributions under varying concurrency, identifying when each pipeline wins on P99/P50 ratio (jitter compression).

The [project/](project/) directory is currently empty — this is greenfield work. The plan below decomposes execution into the three phases the user requested: **(1) model dev & export, (2) servers, (3) benchmark**. Each phase is independently shippable so failures stay contained and the benchmark phase can iterate without re-touching the model.

**Decisions locked:**
- **Dataset:** HIGGS (UCI, 11M rows × 28 numeric features) — dense, purely numeric, eliminates preprocessing as a confounder so latency reflects model execution.
- **Isolation:** single host, CPU pinning via `taskset`/cgroups, disjoint cores for server vs Vegeta, fixed container CPU/memory caps.
- **Repo layout:** `project/{1-model, 2-serving/{fastapi,bentoml}, 3-bench}` — numbered to match thesis phases (Phase 2 contains both serving pipelines).

---

## Phase 1 — Model Development & Export

**Goal:** Produce a single, version-pinned ONNX artifact that both servers will load. The artifact is the contract between phases 1 and 2.

**Steps:**
1. **Data acquisition & split** (`project/1-model/data/`)
   - Download HIGGS from UCI; verify checksum.
   - Standard split: first 10.5M rows train, last 500k test (the canonical HIGGS split — keeps results comparable to the literature).
   - Persist train/test as Parquet for fast reload.
2. **Training** (`project/1-model/train.py`)
   - XGBoost binary classifier, `hist` tree method, fixed `seed`.
   - Target a non-trivial tree count (~500 trees, depth 8) — enough that inference latency is measurable but not so deep that ONNX export becomes the bottleneck.
   - Log training metrics (AUC, accuracy) to a `model_card.md` so the thesis can cite an honest accuracy figure; the thesis is *not* about accuracy, but the model must be defensible.
3. **ONNX export** (`project/1-model/export.py`)
   - Use `onnxmltools.convert_xgboost` with `target_opset` pinned and float32 input tensor `[None, 28]`.
   - Sanity-check parity: run 10k random test rows through both the XGBoost Python predictor and `onnxruntime.InferenceSession`; assert max-abs prob diff < 1e-5.
4. **Artifact handoff** (`project/1-model/artifacts/model.onnx` + `schema.json`)
   - `schema.json` records input dtype/shape, output names, and a SHA256 of the `.onnx` file. Both servers load by SHA — guarantees they serve the *exact* same model.

**Exit criteria:** `model.onnx` exists, parity test passes, model card committed.

---

## Phase 2 — Server Development

**Goal:** Two containerized HTTP services exposing the same `POST /predict` contract, differing only in serving stack. Identical input schema, identical output schema, identical CPU/memory budgets.

**Shared contract:**
- Request: `{"features": [[f1..f28], ...]}` (batchable on the wire, single-row used by default).
- Response: `{"scores": [p1, ...]}` (positive-class probability).
- Both containers: 2 CPU limit, 2 GiB memory limit, ONNX Runtime `intra_op_num_threads=2`, `inter_op_num_threads=1` (locked to container CPU budget per the proposal).

**Pipeline A — FastAPI + ONNX Runtime** (`project/2-serving/fastapi/`)
- `app.py`: FastAPI app, lifespan-loaded `InferenceSession`, `uvicorn` with single worker.
- `Dockerfile`: slim Python base, only `fastapi`, `uvicorn`, `onnxruntime`, `numpy`.
- Represents the "generalist async web framework" arm.

**Pipeline B — BentoML + adaptive batching** (`project/2-serving/bentoml/`)
- `service.py`: BentoML service, ONNX runner, `@bentoml.api(batchable=True)` with `max_batch_size` and `max_latency_ms` configured.
- `bentofile.yaml` + `Dockerfile` produced via `bentoml containerize`.
- Represents the "specialized ML serving with adaptive batching" arm.

**Operational parity checks** (`project/3-bench/parity.py`):
- Send 1000 identical rows to both servers; assert score parity within 1e-5.

**Exit criteria:** Both containers build, pass `/healthz`, agree on scores, and accept Vegeta-shaped POST requests.

---

## Phase 3 — Benchmark

**Goal:** Generate the latency distributions that constitute the thesis's primary deliverable.

**Bench harness layout** (`project/3-bench/`)
- `targets/`: Vegeta target files (one row of HIGGS test features per request).
- `run.sh`: orchestrates a single bench run — pins server to cores 0–1, pins Vegeta to cores 2–3 (`taskset`), warms up, runs measurement, dumps `.bin` results.
- `sweep.py`: drives the matrix, collects results into `results/<pipeline>/<rps>/<concurrency>/`.
- `analyze.ipynb`: reads results, computes P50/P95/P99/P99.9, P99/P50 ratio, RPS achieved, error rate; produces thesis plots.

**Experimental matrix:**
| Axis | Values |
|---|---|
| Pipeline | FastAPI, BentoML |
| (workers, ORT intra-op threads) | (1,1), (1,2), (2,1), (2,2) — see "Parallelism cells" below |
| Vegeta concurrency | 1, 10, 50, 100 |
| Target rate (RPS) | calibrated from saturation pilot, sweep up |
| Duration | 60s measurement after 50–100 req warmup |
| Repeats | 5 runs per cell |

**Parallelism cells (advisor feedback):** the (workers × ORT intra-op threads)
axis was added after advisor review. Each cell isolates a different
concurrency mechanism within the same 2-CPU container budget:

| Configuration | Objective |
|---|---|
| 1 worker / 1 thread | ONNX clean baseline (no concurrency anywhere) |
| 1 worker / 2 threads | ONNX internal parallelism (intra-op) |
| 2 workers / 1 thread | Serving-layer parallelism (process replication) |
| 2 workers / 2 threads | Contention risk (4 thread-slots on 2 cores) |

The same 4 cells are exercised for both pipelines. BentoML's adaptive
batching parameters (`max_batch_size`, `max_latency_ms`) are calibrated once
([project/2-serving/bentoml/calibration_report.md](project/2-serving/bentoml/calibration_report.md))
and held fixed across the matrix so the parallelism axis is the only thing
varying.

**Pre-flight:**
1. Saturation pilot: ramp Vegeta until error rate hockey-sticks.
2. Cold-start isolation run (skips warmup, reported separately).

**Output artifacts:**
- Latency histograms per (pipeline, concurrency, RPS).
- Summary table: P50/P95/P99/(P99/P50) across full matrix.
- Inflection plot: concurrency on x-axis, P99 on y-axis, one line per pipeline.

**Exit criteria:** Results matrix complete, plots generated, raw Vegeta `.bin` dumps stored.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| CPU contention masking jitter | Hard CPU pinning via `taskset` |
| BentoML batching pathology at low RPS | Document as a finding, not a bug |
| HIGGS too compute-bound to discriminate pipelines | Bump tree count to 1000 if pilot shows no difference |
