# 2. Background

## 2.1 Tree-ensemble inference

### 2.1.1 XGBoost as a tabular classifier

<<SOURCE: thesis/phase1.md §"Model" — copy the hyperparameter table and the
1-paragraph justification for HIGGS + XGBoost as the workload.>>

<<TODO: 2 paragraphs on (a) the XGBoost training algorithm at a level
sufficient to motivate the model size used here (500 trees × depth 8 ≈
128k nodes) — cite <<CITE: chen2016>>; (b) why tree ensembles dominate
tabular ML in 2024–2026 — cite <<CITE: grinsztajn2022>>.>>

### 2.1.2 ONNX as a portable model format

<<TODO: 1 paragraph on what ONNX is, why we use it instead of native XGBoost
serialisation (single runtime across both servers; bit-identical scores
across pipelines — verified by parity gate at 2.98e-7 max abs diff). Cite
<<CITE: onnx>>.>>

### 2.1.3 ONNX Runtime's TreeEnsembleClassifier operator

<<TODO: 2 paragraphs on (a) the TreeEnsembleClassifier operator
implementation — sequential tree traversal with branch prediction +
prefetching, no SIMD for tree paths — and (b) the theoretical floor for
single-row inference at 500 trees × depth 8 (≈ 4000 comparisons; on a 3-4
GHz CPU with hot cache and good branch prediction, ~14-25 µs is the
predicted floor). Cite the ORT documentation and any TreeEnsemble paper. See
`thesis/monograph/l1_literature_check.md` for the corroboration.>>

## 2.2 Tail latency and percentile statistics

### 2.2.1 Why P99, not mean

<<TODO: 2 paragraphs on the Tail-at-Scale phenomenon — cite
<<CITE: deans>> — and why for user-facing services the right operational
metric is P99 (often P99.9). One concrete user-facing example.>>

### 2.2.2 The CCDF view

<<TODO: 1 paragraph on the complementary cumulative distribution function
P(latency > t) as the right shape to inspect tails, since percentile-tables
can hide bimodality or heavy tails. Reference the CCDF plot we produce
(`results/figures/ccdf_tail.png`).>>

### 2.2.3 Jitter as P99/P50

<<TODO: 1 paragraph defining jitter as the ratio P99/P50 (tail variability
normalised against the median) and why a ratio rather than an absolute
difference is the right shape for cross-pipeline comparison.>>

## 2.3 Open-loop vs closed-loop load testing

### 2.3.1 Closed-loop testing and Coordinated Omission

<<TODO: 2 paragraphs on Gil Tene's wrk2 + Coordinated Omission analysis — a
closed-loop generator (each next request waits for the prior response) under-
reports tail latency because slow responses suppress arrival rate during
exactly the period of interest. Cite <<CITE: tene>>.>>

### 2.3.2 Open-loop testing with Vegeta

<<TODO: 1 paragraph explaining Vegeta's open-loop design: at a fixed rate R
RPS, requests arrive every 1/R seconds regardless of in-flight count.
Backpressure surfaces as growing queue depth (and thus latency or 503s),
not as suppressed arrival rate.>>

### 2.3.3 Why this matters for an ML serving comparison

<<TODO: 1 paragraph stating: any benchmark that uses `wrk`, `ab`, or other
closed-loop generators to compare serving stacks systematically
underestimates the tail of whichever stack queues more. Vegeta lets the
queueing-vs-shedding distinction surface honestly.>>

## 2.4 Queueing theory primer

### 2.4.1 M/M/c as a first-order model

<<TODO: 2 paragraphs introducing M/M/c queues at the level needed to
predict where saturation should appear: utilisation ρ = λ/(cμ); for ρ → 1
the queue depth grows unboundedly. Apply to our setup: c = number of
parallel inference paths (worker × thread for FastAPI; 1 dispatcher
coroutine per worker for BentoML); μ derived from L1 measurement.>>

### 2.4.2 Little's Law

<<TODO: 1 paragraph stating Little's Law (L = λW) and applying it to
estimate in-flight request count from RPS × mean latency. Reference the
`little_law_n_in_flight` column we added to `summary.csv`.>>

### 2.4.3 Load-shedding vs queueing systems

<<TODO: 1 paragraph distinguishing systems that absorb overload as latency
(FastAPI) from systems that shed load via 503 when a deadline is missed
(BentoML's dispatcher). This motivates the goodput-within-SLA metric in
Chapter 4.>>

## 2.5 ML serving stack components

### 2.5.1 FastAPI + uvicorn

<<TODO: 2 paragraphs on (a) FastAPI's design — Pydantic + Starlette + asyncio
event loop; (b) uvicorn's worker model — master process forks N workers
sharing a listening socket; per-connection load balancing. Forward-reference
the worker-axis dormancy finding in Chapter 6.>>

### 2.5.2 ONNX Runtime in a Python web handler

<<TODO: 1 paragraph on the critical detail: `session.run` releases the GIL
during native inference, so a sync FastAPI handler can be dispatched to
the AnyIO threadpool and parallelise — while an async handler would
serialise on the single event loop. This is non-obvious and is the
implementation hinge for FastAPI's performance.>>

### 2.5.3 BentoML

<<TODO: 2 paragraphs on (a) BentoML 1.4's decorator-based service API —
`@bentoml.service`, `@bentoml.api(batchable=True, batch_dim=0)`; (b) the
adaptive batching dispatcher's design — a single asyncio `controller()`
coroutine per worker that collects requests, decides when to dispatch a
batch, and routes results back. Cite the BentoML docs and the dispatcher
source <<CITE: bentoml_dispatcher>>.>>

## 2.6 Adaptive batching theory

### 2.6.1 Where batching pays off

<<TODO: 2 paragraphs on the cost model: per-request batching adds queueing
delay τ_wait but amortises fixed per-call overhead k across batch size B.
Net benefit ≈ k(1 − 1/B) − τ_wait. For deep networks and GPU kernels, k is
large (kernel launch + Python boilerplate + tensor materialisation), so
B = 16 buys you a ~16× per-call cost reduction at the cost of one τ_wait.
For tree ensembles on CPU, k ≈ inference cost itself, so the trade
inverts.>>

### 2.6.2 `max_latency_ms`: a deadline, not a wait window

<<TODO: 1 paragraph explaining that BentoML's `max_latency_ms` is the
hard request-completion deadline used to compute whether to drop a queued
request (513-line dispatcher.py — exact citation in §5). The original
calibration procedure misread this as a wait window to minimise, producing
the lat5 variant that systematically sheds load; the Phase 3 measurement
correction promotes this to an explicit axis with `lat50` and `lat250`
variants.>>

## 2.7 Statistical methods

### 2.7.1 Bootstrap confidence intervals on percentiles

<<TODO: 1 paragraph on percentile bootstrap — resample each cell's repeats
with replacement, recompute the percentile, take 2.5 / 97.5 quantiles of
the bootstrap distribution. Used in `results/ci.csv`. Cite
<<CITE: efron1979>>.>>

### 2.7.2 Mann–Whitney U and Cliff's δ

<<TODO: 1 paragraph on non-parametric significance testing (Mann–Whitney U)
and an effect-size measure that is invariant to scale (Cliff's δ), used in
`results/effect_sizes.csv` to compare pipelines pairwise per cell. Cite a
standard reference.>>

### 2.7.3 Coefficient of variation across repeats

<<TODO: 1 paragraph defining CoV = σ/μ across the 3 repeats of each cell;
used as the run-to-run stability check before any pipeline-to-pipeline
comparison.>>
