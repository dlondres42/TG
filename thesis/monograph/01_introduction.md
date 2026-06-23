# 1. Introduction

## 1.1 Context and motivation

Modern web-tier ML serving is dominated by two design philosophies. The first
treats each incoming request as an independent unit of work and parallelises
inference across worker processes and threads; FastAPI with ONNX Runtime is a
canonical instance of this design. The second introduces a request-batching
layer between the network handler and the inference engine to amortise
per-call overhead across concurrent requests; BentoML with adaptive batching is
a canonical instance. The two philosophies make different bets about *where the
cost of an ML request lives*.

<<TODO: write 2 paragraphs on (a) why tail latency (P99) is the metric of
interest for online ML services — cite the standard Tail-at-Scale paper
<<CITE: deans>>, the user-facing impact, and the difference between
average-case and tail-case operational behavior; (b) why XGBoost on tabular
data is the workload that exposes the trade-off most starkly — cite
<<CITE: grinsztajn2022>> and <<CITE: chen2016>>.>>

## 1.2 Problem statement

For a *cheap* tabular workload — single-row XGBoost classification on dense
float32 features, with per-inference cost in the microsecond range — does
adaptive batching pay off, or does its per-request plumbing cost exceed any
batching amortisation? The proposal hypothesised that adaptive batching would
absorb concurrency more gracefully and reduce tail latency under high RPS; the
results contradict that hypothesis and the mechanism is identifiable.

<<TODO: 1 paragraph stating the research question crisply, ~3 sentences.>>

## 1.3 Contributions

This monograph contributes:

1. **A quantitative latency comparison of FastAPI + ONNX Runtime and BentoML +
   adaptive batching on identical hardware, model artifact, and request
   contract**, sweeping a 219-cell matrix of (worker × thread × RPS) under
   open-loop load with three BentoML deadline variants.
2. **A three-layer latency-budget decomposition** isolating the model's
   inference cost (L1), the framework + HTTP cost on the host (L2), and the
   Docker overhead (L3), enabling attribution rather than just totals.
3. **A fairness-corrected metric — goodput within an SLA** — that lets a
   load-shedding system (BentoML, which returns 503 when the deadline is
   missed) and a queueing system (FastAPI, which absorbs overload as latency)
   be compared on the same footing.
4. **Direct CPU-utilisation evidence for the dispatcher mechanism**: at
   identical (1-worker, 1-thread, 600 RPS) load, BentoML burns ~2.5× the CPU
   as FastAPI and still collapses, fingerprinting one asyncio coroutine
   pinned to one core per worker as the bottleneck.
5. **A reproducibility package** comprising the SHA-pinned ONNX artifact, the
   per-cell `version.json` capturing the live applied parallelism knobs, the
   raw Vegeta binaries, and a containerised bench harness.

## 1.4 Findings preview

<<TODO: 1 paragraph stating the headline findings — FastAPI sustains 99.7-100%
goodput-within-50-ms-SLA across the matrix; BentoML's P99 climbs by ~3 orders
of magnitude at the same offered load; adaptive batching adds no measurable
benefit at this scale because inference is too cheap to amortise; the
dispatcher coroutine, not lack of CPU, is the bottleneck — and a sentence
saying when the BentoML approach would be expected to win (GPU, deep
networks, models with significant fixed per-call overhead).>>

## 1.5 Organisation

Chapter 2 (*Background*) reviews tree-ensemble inference, ONNX Runtime's
operator implementation, open-loop benchmarking and the Coordinated Omission
problem, and adaptive batching theory. Chapter 3 (*Related work*) surveys
prior ML-serving comparisons and decision-forest inference benchmarks.
Chapter 4 (*Methodology*) details the experimental design: the three-layer
decomposition, the matrix sweep, the CPU pinning protocol, the open-loop
load generator, and the statistical procedures. Chapter 5 (*Implementation*)
documents the two servers, the bench harness, and the reproducibility
contract. Chapter 6 (*Results*) presents the data: L1 floor, L2 host HTTP,
the headline L3 matrix, the CPU mechanism evidence, and the statistical
confidence intervals. Chapter 7 (*Discussion*) interprets the results,
addresses threats to validity, and identifies the regimes in which the BentoML
design *would* be expected to win. Chapter 8 (*Conclusion*) summarises and
points to future work.
