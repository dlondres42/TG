# Part 2 — Serving Pipelines and Phase 3 Plan

**TG: P99 Tail Latency for Tabular ML Serving**
*David Londres — CIn/UFPE*
*Date: 2026-05-22*

## Context

Part 2 produces two containerised HTTP services that load the Phase 1 artifact
`model.onnx` and expose the same `POST /predict` contract. The pipelines differ
only in the serving stack under comparison: **(A) FastAPI + ONNX Runtime**,
representing a generalist async web framework with no batching, and **(B)
BentoML 1.4 + ONNX Runtime with adaptive batching**, representing a specialised
ML-serving framework. Both containers share an identical CPU/memory budget
(2 CPU, 2 GiB) and verify the SHA-256 of `model.onnx` against `schema.json`
at startup, so any difference observed in Phase 3 is attributable to the
stack, not to the model.

## Pipeline A — FastAPI + ONNX Runtime

The handler is intentionally synchronous (`def`, not `async def`) so that
ONNX inferences run in FastAPI's threadpool and concurrent requests interact
with the worker × ORT-thread matrix as intended. Wrapping ONNX in an `async
def` handler would serialise every request behind one event-loop task and
collapse the worker axis into an asyncio artifact. There is no request
batching: single-request inference is this pipeline's identity. The AnyIO
sync-handler threadpool that Starlette dispatches into (default 40 tokens
per worker) is pinned explicitly to 32 via `FASTAPI_THREADPOOL_SIZE` and
recorded in `/version`, so it cannot drift between runs; the in-flight
count is verified to stay well below this cap at the Phase 3 RPS levels.

## Pipeline B — BentoML + adaptive batching

The service uses BentoML 1.4's decorator API with `@bentoml.api(batchable=True,
batch_dim=0)`. At the wire level, the dispatcher collects concurrent requests,
stacks their `features` arrays along axis 0, calls inference once on the
combined batch, then splits the output back per request. The batching
behaviour is governed by two knobs read from `batching.json`:
`max_batch_size` and `max_latency_ms`.

## Batching calibration

The two batching knobs trade throughput against queueing wait, with no
universal optimum. To keep the Phase 3 comparison "best vs best", a small
3 × 3 grid was swept under fixed parallelism (1 worker / 2 ORT threads) at
100 RPS open-loop for 15 s per cell. The pick was the cell with the lowest
P99 among zero-error cells, tie-breaking on P50: **`max_batch_size = 16`,
`max_latency_ms = 5`**, P99 = 35.4 ms, no errors. Tighter latency windows
(`= 2 ms`) produced dispatcher overload (~3% 503s); larger windows (`= 10 ms`)
inflated P99 without benefit. The full grid is in `calibration_report.md`.
These values are now frozen for the entire Phase 3 matrix, so the parallelism
axis remains the only thing varying.

## Parity verification

A 3-way score parity test (FastAPI vs BentoML vs in-process ONNX Runtime
as ground truth) was run with both containers up simultaneously, single-row
requests, 500 rows. Both servers agreed bit-identically with each other
(max-abs diff = 0.00) and matched the local ONNX reference to within
**2.98 × 10⁻⁷**, three orders of magnitude tighter than the 1 × 10⁻⁵
tolerance adopted in Phase 1.

## Phase 3 — what will be evaluated

Phase 3 has two deliverables: a **matrix sweep** of latency under load, and a
**latency budget** that decomposes total latency into stack layers. Both use
open-loop Vegeta load (per the proposal's anti-Coordinated-Omission
requirement) on a single host with disjoint CPU pinning (server cores 0–1,
Vegeta cores 2–3).

### Deliverable 1 — matrix sweep

For both pipelines, P50/P95/P99/P99.9 and the P99/P50 jitter ratio are measured
across the parallelism matrix, swept over request rate:

| Axis | Values |
| ---- | ------ |
| Pipeline | FastAPI, BentoML |
| (workers, ORT intra-op threads) | (1,1), (1,2), (2,1), (2,2) |
| Target rate (RPS) | 5 rates per cell, calibrated from a saturation pilot |
| Repeats | 3 |

Here *workers* is the number of independent server processes (`uvicorn
--workers` / `@bentoml.service(workers=…)`), and *threads* is ORT's intra-op
pool inside one inference call. The runtime — not application code — forks the
workers; this was verified with `docker top` at 2 workers (uvicorn spawns 2
worker processes sharing the socket; BentoML spawns 2 `worker-id` service
processes). Each worker loads its **own** ONNX session (confirmed: BentoML
logged `initialized` once per worker; ~400–480 MiB resident for two model
copies, well under the 2 GiB cap). The four cells isolate distinct ways of
spending the 2-CPU budget: (1,1) clean ONNX baseline, (1,2) ORT internal
parallelism, (2,1) serving-layer parallelism via process replication, (2,2)
deliberate oversubscription to expose contention.

BentoML's batching stays frozen at the calibrated `(16, 5)`, so the
parallelism axis is the only thing varying. One caveat follows from the worker
model: **each BentoML worker runs its own batch dispatcher**, so the 2-worker
cells split the incoming rate across two independent queues — each sees ~half
the RPS and forms smaller batches than the 1-worker cell the calibration was
tuned on. Batching is held fixed deliberately (to isolate parallelism), so the
2-worker BentoML cells run a config that is fixed but not re-optimal; this is
reported as a known limitation rather than hidden. The headline output is the
**inflection plot** — RPS on the x-axis, P99 on the y-axis, one line per
pipeline — whose crossover identifies the load at which each pipeline wins on
tail-latency stability.

### Deliverable 2 — latency budget (layer decomposition)

To attribute the FastAPI-vs-BentoML difference to specific stack components
rather than only reporting totals, the same request is measured through three
layers, all routed via HTTP so BentoML's batching is exercised identically
wherever it applies:

| Layer | Isolates | Method |
| ----- | -------- | ------ |
| **L1 — inference floor** | the model's intrinsic ONNX cost (identical for both pipelines) | tight `session.run` loop, no HTTP |
| **L2 — host HTTP** | framework + request parsing + TCP loopback | server run on the host (no Docker), Vegeta sidecar; one (1,2) cell per pipeline |
| **L3 — containerised HTTP** | + Docker port-mapping / isolation | the full matrix sweep above |

The differences attribute the milliseconds: `L1` is the floor, `L2 − L1` is
framework + HTTP overhead (where BentoML's dispatcher wait-window surfaces),
and `L3 − L2` is server-side Docker overhead. This yields a **stacked-bar
budget chart** (one bar per pipeline, split into the three layers) that turns
"BentoML is N ms slower" into "BentoML spends N ms in the dispatcher in
exchange for higher throughput under load." Cold-start (first request vs warmed
service) is reported separately as a pre-flight pilot. Together the inflection
plot and the budget chart are the thesis's primary quantitative results.
