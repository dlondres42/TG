# 7. Discussion

## 7.1 Why BentoML does not help for cheap tabular inference

<<TODO: 2 paragraphs. The cost model from §2.6 makes the prediction
explicit. Adaptive batching amortises fixed per-call overhead k across
batch size B: net benefit ≈ k(1 − 1/B) − τ_wait. For our model k ≈ 14 µs,
and the dispatcher's plumbing cost (queue ops, async future bookkeeping,
batch assembly) is ~100–1000 µs per dispatch decision. The trade inverts:
batching costs more than it saves. The measurement confirms the
prediction quantitatively — CPU goes into plumbing, not inference.>>

## 7.2 When BentoML would win

<<TODO: 2 paragraphs identifying the workloads where the comparison would
flip. (a) GPU inference: kernel launch + tensor materialisation costs
~50–500 µs of fixed overhead, so batching 16 saves ~750–7500 µs at the
cost of one τ_wait. (b) Deep networks: per-inference cost is in the
ms range, so the dispatcher's overhead is small relative to inference.
(c) Workloads with large per-call deserialisation: same logic. State the
boundary explicitly — for inference cost k > ~1 ms, BentoML's design
starts to pay off.>>

## 7.3 The dispatcher coroutine as a generalisable observation

<<TODO: 2 paragraphs. The single-event-loop dispatcher pattern is not
unique to BentoML — Triton's dynamic batcher, TF-Serving's batching
options, custom Python services around vLLM all have similar
architectures. The thesis observation — that for sub-millisecond inference
the dispatcher's serialised plumbing dominates — likely generalises. State
this carefully (we have not measured Triton/TFS); flag it as a hypothesis
testable by future work.>>

## 7.4 The deadline-knob distinction

<<TODO: 2 paragraphs. The Phase 3 measurement correction (`max_latency_ms`
promoted from frozen knob to explicit axis) revealed that ~80% of the
"BentoML collapses 5–10× earlier" effect observed in the first sweep was a
*configuration artifact* of the original calibration, not an
architectural property. The configuration knob is a deadline; the
calibration mis-interpreted it as a wait-window-to-minimise. This is a
methodological lesson worth stating clearly: when comparing systems with
configurable knobs that affect operational behaviour, the knobs must be
swept as axes rather than frozen "best" values, or the comparison can be
unintentionally rigged against the load-shedding system.>>

## 7.5 The L1 figure: literature corroboration

<<SOURCE: thesis/monograph/l1_literature_check.md.>>

<<TODO: 2 paragraphs. The 14 µs single-row L1 floor is consistent with
the lleaves benchmark (ONNX Runtime at 11 µs on older Haswell hardware)
and brackets the Asadi–Lin (2014) theoretical floor of 8–20 µs for 500
trees × depth 8 on a modern CPU. The often-cited Hummingbird "~0.2 ms"
figure is the GEMM-tensorisation path on a larger model and does not
bound the native `ai.onnx.ml.TreeEnsembleClassifier` operator the export
uses. Defending the L1 number requires citing the protocol (hot cache,
pure `perf_counter_ns(session.run)`, identical ORT session options as
the servers) and reporting L1 jointly with L2/L3 so the audience sees
that L1 is <2 % of full-stack latency.>>

## 7.6 Threats to validity revisited

### 7.6.1 The FastAPI worker-axis caveat

<<TODO: 1 paragraph. Re-summarise the worker-dormancy finding (§4.8.1
and §6.5.3) and explain why it does *not* invalidate the FastAPI-vs-
BentoML headline. State the planned follow-up explicitly: re-run with
`vegeta attack -keepalive=false` to force per-request TCP setup, so the
kernel's `accept()` spreads load across both uvicorn workers. Expected
outcome: w2 CPU ≈ 2× w1 CPU, w2 P99 improves slightly.>>

### 7.6.2 Single-host scope

<<TODO: 1 paragraph. The proposal explicitly scoped the comparison to
single-host with CPU pinning. The findings should generalise to multi-host
deployments behind a load balancer where each backend host runs at the
same per-host load, but multi-host introduces network hops and L7
distribution that are out of scope here. Multi-host is identified as
future work.>>

### 7.6.3 Workload representativeness

<<TODO: 1 paragraph. HIGGS is dense numeric and balanced. The findings
likely generalise to similar dense numeric tabular workloads but may not
generalise to mixed-type tabular data with categorical preprocessing or
to image / text models with significant per-request decoding. Future work
should re-run on a categorical-feature workload (KDD'09, Criteo) to
test this boundary.>>

## 7.7 Implications for practice

<<TODO: 2 paragraphs of practical recommendations. (a) For cheap tabular
inference on CPU, deploy FastAPI + ONNX Runtime with sync `def` handlers
and `intra_op_num_threads >= 2`; do *not* add adaptive batching unless
inference cost exceeds ~1 ms. (b) When configuring BentoML, treat
`max_latency_ms` as a hard SLA deadline (which is what it is) rather than
a wait window; set it to your service's actual tolerance, not to a
calibrated "best" value at low load. (c) Always validate serving stack
benchmarks with open-loop generators — Vegeta or k6 — to forestall
Coordinated Omission artefacts.>>

## 7.8 Limitations

- n=3 repeats per cell → bootstrap CIs are wide; we rely on effect-size
  rather than tight intervals.
- 219 cells of L3 data plus 60 cells of CPU-mechanism data — sufficient
  for the headline claims but not for fine-grained sensitivity studies
  across additional axes.
- Seed sensitivity (re-train with different seeds) and model-snapshot
  sensitivity were not tested. The SHA-pinned single artifact is the
  reproducibility contract, but the *robustness* of the comparison to
  model retraining is an open question.
- The host OS and Docker runtime versions are pinned in the appendix; we
  did not test on alternative kernels or runtimes.
