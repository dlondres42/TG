# 4. Methodology

## 4.1 Experimental design overview

The comparison sweeps two pipelines (FastAPI + ONNX Runtime; BentoML +
adaptive batching, in three deadline variants) across a parallelism matrix
× RPS axis under open-loop load, decomposing each pipeline's latency into
three measurement layers and capturing both client-side latency and
server-side CPU utilisation. The design isolates the comparison axis
(pipeline) by holding every confounder constant: identical model artifact
(SHA-pinned), identical CPU pinning, identical container resource limits,
identical request contract, identical load generator.

<<SOURCE: PLAN.md §3 — copy and polish the experimental-design overview.>>

## 4.2 The three-layer latency decomposition

<<SOURCE: thesis/phase3.md §"Three-layer decomposition" — copy and edit.>>

| Layer | Isolates | How |
| --- | --- | --- |
| **L1 — Inference floor** | pure ONNX `session.run` cost | tight Python loop, 10 000 iterations, no HTTP |
| **L2 — Host HTTP** | framework + HTTP loopback, no Docker | server pinned via `taskset -c 0,1`, Vegeta hits directly |
| **L3 — Containerised HTTP** | full deployment | server in Docker with `--cpuset-cpus=0,1`, Vegeta native pinned to cores 2–3 |

Attribution: `L1` is the inference floor (one number across all variants);
`L2 − L1` is the framework + asyncio + uvicorn/BentoML routing + TCP loopback
cost; `L3 − L2` is the Docker port-mapping and container isolation overhead.

<<FIG: results/figures/budget_stacked_bar.png>>

<<TODO: 1 paragraph on what the decomposition cannot isolate (sysmon /
scheduler noise; Linux kernel TCP backlog tuning) and why those are bounded
by the noise-floor measurement (§4.5.4).>>

## 4.3 The L3 matrix sweep

### 4.3.1 Axes

<<SOURCE: PLAN.md §3.5.1, project/3-bench/layers/L3_docker_sweep.py.>>

| Axis | Values | Cardinality |
| --- | --- | ---: |
| Pipeline variant | `fastapi`, `bentoml_lat5`, `bentoml_lat50`, `bentoml_lat250` | 4 |
| (workers, ORT threads) | (1,1), (1,2), (2,1), (2,2) | 4 |
| Target RPS | {50, 150, 350, 600, 1000} (early-abort cap) | up to 5 |
| Repeats | 3 | 3 |

Resulting in 219 completed cells out of 240 nominal (collapse-aborted cells
beyond the per-(variant, w, t) saturation point are skipped to bound wall
time).

### 4.3.2 Variant axis — `max_latency_ms` promoted from a fixed knob

<<TODO: 2 paragraphs on the Phase 3 correction. The original calibration
froze `max_latency_ms=5` as the "best" config (lowest P99 at low load); the
dispatcher source revealed that this value is a *hard request deadline*,
not a wait window, and 5 ms is too tight for the L2 floor (~3-4 ms). We
promoted the knob to an explicit axis (lat5 / lat50 / lat250) to separate
configuration artifact from architectural behaviour. See §5 for the
implementation; see Chapter 6 for the results.>>

## 4.4 Open-loop load with Vegeta

<<TODO: 2 paragraphs explaining the open-loop generator setup. Each cell
runs at a fixed rate R for 45 seconds; Vegeta opens connections as needed
and dispatches requests at 1/R intervals regardless of in-flight count.
Per-request timeout 3 s (caps connection-pool hangs on saturated servers).
Cite the proposal §3.3 anti-Coordinated-Omission requirement and Tene's
wrk2 analysis.>>

On WSL2 Linux, Vegeta runs natively (`taskset -c 2,3 vegeta attack`) for
the cleanest isolation. The Windows fallback path uses a Docker sidecar
container also pinned to cores 2–3 (`--cpuset-cpus=2,3`); the route was
validated during the Phase 3 migration. All Phase 3 headline results use
the native Linux path.

## 4.5 Statistical procedures

### 4.5.1 Repeats and cell-level aggregation

<<TODO: 1 paragraph: each cell is repeated 3 times; per-cell percentiles
are the median across repeats; bootstrap CIs are computed from the 3
repeats with 10 000 bootstrap samples (`stats.bootstrap_ci`).>>

### 4.5.2 Bootstrap confidence intervals on percentiles

<<TODO: 1 paragraph on the bootstrap percentile method (Efron 1979). We
report 95% CIs on P50, P95, P99 per cell in `results/ci.csv`. At n=3
repeats, CIs are wide; we use them as a *plausibility check* rather than a
tight bound, and rely on the size of the FastAPI-vs-BentoML gap (orders
of magnitude in P99) for inferential weight.>>

### 4.5.3 Non-parametric significance and effect size

<<TODO: 1 paragraph: Mann–Whitney U test for pipeline-vs-pipeline at each
cell, Cliff's δ as a scale-invariant effect size. Both in
`results/effect_sizes.csv`. We report effect size as the primary inferential
quantity because the sample sizes (cell × repeat) make p-values
uninformative.>>

### 4.5.4 Noise floor

<<TODO: 1 paragraph: L0 noise floor probe measures `/healthz` at 1 RPS for
60 s. Result: FastAPI 1.18 ms P50; BentoML 1.60 ms P50. This is the upper
bound on per-request transport overhead at low load, not a floor on steady-
state latency (the latter is amortised by Vegeta's keep-alive connection
pool under sustained load). Cite the dedicated probe and explain the
amortisation effect.>>

## 4.6 CPU pinning protocol

<<SOURCE: PLAN.md §3.7 — copy the pinning table.>>

| Layer | Server pinning | Vegeta pinning |
| --- | --- | --- |
| L1 | n/a (no server) | n/a (no Vegeta) |
| L2 | `taskset -c 0,1` on host | native Vegeta with `taskset -c 2,3` |
| L3 | Docker `--cpuset-cpus=0,1 --cpus=2` | native Vegeta with `taskset -c 2,3` |

Cores 4–11 are idle; Linux scheduler can still place kernel work there,
but the disjoint pinning between server (0–1) and load generator (2–3)
keeps cross-contamination out of the measurement path.

## 4.7 Server-side CPU sampling

<<TODO: 2 paragraphs on the CPU sampler design. `docker stats --no-stream`
polled at ~1 Hz (the native polling interval of `docker stats`) while the
Vegeta attack runs; per-cell `cpu_pct.csv` captures (timestamp, cpu_pct,
mem_mb). For 20 mechanism-critical cells we re-ran with this sampler
active (the headline 219-cell dataset is intact; CPU runs are in
`results/layer3_with_cpu/`). The 1 Hz density gives ~22 samples per 45 s
cell — sufficient for mean and P95, marginal for sub-second jitter.>>

## 4.8 Threats to validity

### 4.8.1 FastAPI worker-axis dormancy

The 3-bench harness drives Vegeta with HTTP/1.1 keep-alive on by default.
At sustained sub-millisecond P50 RPS, Vegeta's connection pool collapses to
one or two long-lived TCP connections, each pinned to a single uvicorn
worker's accept queue. The result is that FastAPI w2 cells run effectively
as w1 cells from the load balancer's perspective. This is confirmed
empirically by the CPU mechanism subset: w1t1@600 and w2t1@600 have
indistinguishable mean CPU (~38 %). The headline FastAPI-vs-BentoML
comparison is not affected because the relevant comparison is per-cell.
The FastAPI *worker-axis* comparison within the headline data is degenerate
and is reported as such; a future no-keepalive follow-up (Chapter 8) would
close this gap.

### 4.8.2 BentoML dispatcher train-time transient

BentoML's adaptive batching dispatcher has a 30-second `train_optimizer`
warmup during which the batch-size estimator stabilises. The warmup phase
of the harness was bumped from 30 to 100 requests after observing
inflated error rates on `run1` of some cells; re-runs with the longer
warmup are clean. All Phase 3 headline cells use WARMUP=100.

### 4.8.3 Vegeta as a Docker sidecar (Windows path) vs native (WSL2)

The original Windows harness ran Vegeta as a Docker sidecar. On WSL2 we
migrated to native Vegeta with `taskset` for cleaner isolation. Headline
results use the native path; the Windows sidecar artifacts are preserved at
`results/results_windows_baseline/` for cross-platform sanity checks (the
collapse boundaries shift slightly but the qualitative findings are
identical).

### 4.8.4 Single-host setup

The proposal explicitly scopes the comparison to a single-host setup with
CPU pinning. A multi-host extension is identified as future work.

### 4.8.5 HIGGS as a representative tabular workload

<<TODO: 1 paragraph: HIGGS is purely numeric, dense, and balanced. The
results generalise to similar dense numeric tabular workloads (financial,
sensor, network) but may not generalise to mixed-type tabular data with
categorical features, since preprocessing changes the per-request shape.
Cite <<CITE: grinsztajn2022>> on tree ensembles' suitability for tabular
data.>>
