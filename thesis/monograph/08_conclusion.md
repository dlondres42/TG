# 8. Conclusion

## 8.1 Summary

This monograph compared two serving stacks — FastAPI + ONNX Runtime, and
BentoML + adaptive batching — for a cheap tabular ML workload: single-row
XGBoost classification on 28 dense float32 features. The comparison swept a
219-cell matrix of (worker × thread × RPS) under open-loop Vegeta load,
decomposed each pipeline's latency into three measurement layers (pure
inference, host HTTP, containerised), promoted BentoML's `max_latency_ms`
to an explicit axis after discovering that the calibrated 5 ms value was a
configuration artifact, and captured server-side CPU utilisation on a 20-cell
mechanism-critical subset.

## 8.2 Findings

<<TODO: 1 paragraph re-stating the findings: FastAPI dominates the matrix on
goodput-within-SLA (99.7–100 % across the entire matrix), P99 (flat near
2–3 ms vs BentoML's 5 → 3000 ms across an RPS sweep), and jitter (2–4×
vs BentoML's 100–300×). The mechanism — direct CPU evidence — is that
BentoML's per-worker dispatcher coroutine pins a single core regardless of
thread budget, while FastAPI parallelises across ORT threads. Adaptive
batching adds no measurable benefit at this scale because the model's 14 µs
single-row inference cost is well below the dispatcher's per-request
plumbing overhead. The result is expected to flip for inference cost
exceeding ~1 ms (GPU workloads; deep networks).>>

## 8.3 Contributions

<<SOURCE: Chapter 1 contributions list.>>

1. A quantitative latency comparison of two serving stacks on an
   identical request contract, model artifact, and CPU pinning.
2. A three-layer latency-budget decomposition that attributes overhead
   to specific stack components.
3. A fairness-corrected metric — goodput within an SLA — for comparing
   load-shedding to queueing systems.
4. Direct CPU-utilisation evidence for the dispatcher-coroutine bottleneck
   in BentoML.
5. A reproducibility package: SHA-pinned artifacts, per-cell `version.json`,
   raw Vegeta binaries, containerised harness.

## 8.4 Methodological lessons

<<TODO: 2 paragraphs. (a) When comparing systems with operational knobs
that affect behaviour at the comparison boundary — like BentoML's
`max_latency_ms` — the knobs must be swept as axes rather than frozen at
"best" values selected at low load. (b) Open-loop generators are necessary
when comparing queueing-vs-shedding systems; closed-loop generators
systematically under-report tails of whichever system queues more.>>

## 8.5 Future work

### 8.5.1 Closing the FastAPI worker-axis caveat

A small follow-up — re-running FastAPI w2* cells with
`vegeta attack -keepalive=false` to force per-request TCP setup — would
quantify the keep-alive worker-dormancy effect and let the FastAPI
worker-axis comparison be reported on equal footing with BentoML's. ~15
minutes of bench time; included in the run plan
(`thesis/phase3_cpu_mechanism_plan.md` §6) and not yet executed.

### 8.5.2 Extending to GPU and deep networks

<<TODO: 1 paragraph. The cost model in §7.1 predicts the comparison
flips when inference cost exceeds ~1 ms. A re-run with a deep network
(ResNet-18, BERT-small) on GPU would test this prediction empirically and
extend the thesis's claims to the workload class for which adaptive
batching was designed.>>

### 8.5.3 Multi-host comparison

<<TODO: 1 paragraph. Single-host scope leaves out L7 load balancing and
network hops. A multi-host comparison with a real reverse proxy
(nginx/HAProxy) would test whether the FastAPI worker-axis dormancy
generalises to production-realistic deployments.>>

### 8.5.4 Categorical / mixed-type tabular workloads

<<TODO: 1 paragraph. HIGGS is dense numeric. A categorical-feature
workload (KDD'09 or Criteo) introduces per-request decoding that may
change the cost balance between the two pipelines. Worth testing as a
boundary condition.>>

### 8.5.5 Seed and model-snapshot sensitivity

<<TODO: 1 paragraph. The single-artifact reproducibility contract proves
*the experiment* is repeatable but does not prove *the comparison* is
robust to model retraining. A small sensitivity sweep over (seed,
n_trees, max_depth) would quantify the size of the effect.>>

## 8.6 Closing

<<TODO: 1 paragraph closing the monograph. State plainly: for cheap
tabular inference on CPU, FastAPI + ONNX Runtime is the right default.
Adaptive batching is the right tool for the workloads it was designed
for — large fixed per-call overhead — not for the workloads where its
plumbing cost dominates. The mechanism is identifiable; the comparison
is reproducible; the conclusion is operational.>>
