# 3. Related work

## 3.1 Decision-forest inference platforms

<<TODO: 2-3 paragraphs surveying tree-ensemble inference platforms — XGBoost
native, LightGBM, ONNX Runtime, Treelite, Hummingbird, lleaves, RapidScorer,
QuickScorer. Anchor on published per-row latencies; the lleaves benchmark
(11 µs ORT vs 9.6 µs lleaves vs 28 µs Treelite vs 52 µs LightGBM on the
same hardware) is the strongest direct comparison. See
`l1_literature_check.md` for the source list.>>

<<CITE: nakandala2020>> (Hummingbird, OSDI'20) — most-cited tree-inference
benchmark; reports the GEMM-tensorisation path numbers, not the native
operator, so its "~0.2 ms" figure is sometimes misinterpreted as a bound on
ORT's TreeEnsembleClassifier. Distinguishes our work because we use the
native operator and report its actual cost.

<<CITE: asadi2024>> (Decision Forest Platforms from a DB Perspective) —
comprehensive comparison across Treelite, Hummingbird, TF-DF, ONNX. Confirms
ORT is competitive at small batch sizes and that the regime we operate in
(batch-1, low-µs per row) is the published one.

<<TODO: 1 paragraph on what's *missing* from this body of work: none of the
above measure the *serving stack overhead around* tree inference. They
benchmark the inference engine in isolation. This thesis fills that gap.>>

## 3.2 ML serving stack comparisons

### 3.2.1 FastAPI as a serving backbone

<<TODO: 2 paragraphs on prior comparisons involving FastAPI for ML serving.
Cite <<CITE: zaharia2024>> (FastAPI vs Triton on Kubernetes) — the relevant
context is healthcare AI inference, but methodology is closed-loop and
does not isolate latency budget; we improve on this by using open-loop
load and per-layer decomposition.>>

### 3.2.2 BentoML and adaptive batching

<<TODO: 2 paragraphs on BentoML's design (cite docs), the adaptive batching
literature it draws from (Triton Inference Server's dynamic batcher,
TensorFlow Serving's batching options), and the lack of published
microbenchmarks for the dispatcher itself.>>

<<TODO: 1 paragraph distinguishing this work — we measure the dispatcher's
overhead *directly* by sweeping `max_latency_ms` as an axis and capturing
CPU utilisation per cell. No prior published comparison does both.>>

### 3.2.3 Triton, TensorFlow Serving, TorchServe

<<TODO: 1 paragraph briefly placing our comparison within the broader serving
ecosystem, explaining why Triton/TF-Serving/TorchServe are out of scope
(GPU-first design, more invasive deployment, the proposal explicitly scoped
to FastAPI vs BentoML).>>

## 3.3 Tail-latency benchmarking methodology

### 3.3.1 The Coordinated Omission problem

<<TODO: 2 paragraphs on Gil Tene's wrk2 work and the broader Coordinated
Omission analysis — why closed-loop generators systematically under-report
tails when the system queues. Cite <<CITE: tene>>.>>

### 3.3.2 Open-loop generators (Vegeta, k6, hey)

<<TODO: 1 paragraph on the open-loop generator landscape; justify the
choice of Vegeta (JSON-output format that downstream tooling can parse,
explicit rate-based attacks rather than concurrency-based, mature; native
binary on Linux for cleanest isolation).>>

### 3.3.3 Goodput-under-SLA as a fairness metric

<<TODO: 1 paragraph on the load-shedding vs queueing fairness problem (a
system that drops 503s under load looks "worse" on error-rate but may be
"better" on the latency-distribution of served requests). The
goodput-within-SLA metric is the standard fix; we cite it as our
fairness-corrected headline.>>

## 3.4 Where this thesis sits

<<TODO: 1 paragraph closing the chapter: prior work measures inference
engines in isolation or measures serving stacks with closed-loop generators.
We do both honestly — open-loop generator, three-layer decomposition, a
fairness-corrected metric — and we measure the *mechanism* (CPU
utilisation evidence for the dispatcher claim), which is a novel
contribution.>>
