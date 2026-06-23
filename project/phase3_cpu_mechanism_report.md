---
title: "Phase 3 follow-up — CPU mechanism evidence"
author: "David Londres — CIn/UFPE"
---

# Context

The headline Phase 3 report concluded that BentoML's per-worker dispatcher coroutine serialises requests and is the bottleneck. That claim was inferred from the latency pattern (collapse RPS scales with workers, not threads). This follow-up measures the **server-side CPU utilisation** directly to confirm the mechanism: a single asyncio coroutine pinning one core would show ~100% CPU on a 1-worker cell while FastAPI at the same RPS uses far less.

# Method

Re-ran **20 hand-picked cells × 3 repeats × 45 s attack** (total 60 runs), with `docker stats --no-stream` polling the running server container at ~1 Hz, writing one `cpu_pct.csv` per cell. Server config and everything else are identical to the headline L3 sweep — only difference is the side-channel CPU capture. Results live under `results/layer3_with_cpu/`; the headline dataset at `results/layer3/` is untouched.

Variants covered: bentoml_lat250, bentoml_lat5, bentoml_lat50, fastapi. Cell selection brackets the collapse band on BentoML (where the mechanism should be visible) and includes FastAPI reference cells at the same RPS so the comparison is apples-to-apples.

# Finding 1 — Same RPS, same parallelism, very different CPU

![Mean CPU% per (cell × variant) with 95% bootstrap CIs across 3 repeats. Dashed lines mark 1-core (100%) and 2-core (200%) saturation under the `--cpus=2` cap.](3-bench/results/figures/cpu_bar.png){ width=100% }

At **w1t1@600 RPS** — the simplest possible cell — FastAPI uses **33.7% CPU** while BentoML (`lat50`) uses **96.1%**, with P99 of 1.3 ms vs 3001 ms and 0.0% vs 80.7% errors. Same model, same threads, same RPS, same container limits — BentoML burns 2–3× the CPU on dispatcher bookkeeping (queue ops, batch assembly, futures) **and still collapses**. The CPU isn't going into inference; it's going into the per-request plumbing the dispatcher imposes.

# Finding 2 — BentoML's CPU saturates one core, not two

On the (1,*) BentoML cells, mean CPU clusters around 95–100% — close to *one core*, not two. That's the signature of a single coroutine pinned to a single core on the asyncio event loop. `bentoml_lat50` w1t1@350 sits at 69.8% CPU with 236 ms P99 — the dispatcher coroutine is already saturated before the RPS axis collapses. On the (2,*) cells the same pattern plays out at ~2× the CPU (e.g. `bentoml_lat50` w2t1@1000 = 183.9%), matching the doubled dispatcher count and explaining why the per-worker collapse RPS approximately doubles with worker count. The dispatcher coroutine **is** the bottleneck, quantified.

# Finding 3 — FastAPI's CPU tracks the thread axis, not the worker axis

![CPU vs P99 scatter — one point per cell, marker size scales with target RPS. BentoML occupies the upper-right (high CPU, high P99) quadrant at every comparable RPS; FastAPI clusters in the lower-left.](3-bench/results/figures/cpu_vs_p99.png){ width=85% }

FastAPI at w1t1@600 (33.7%) and w2t1@600 (35.9%) are statistically indistinguishable — adding a second uvicorn worker did not change CPU. The thread axis is different: w1t2@600 (124.5%) is roughly 3× the w1t1 value, matching ORT's second intra-op thread becoming active. This is the **HTTP/1.1 keep-alive worker-dormancy** effect we'd flagged: Vegeta's small connection pool, at sustained sub-millisecond RPS, pins to one worker's accept queue, so the second worker stays idle. The mechanism comparison (FastAPI vs BentoML at the same (1,*) cell) is unaffected by this — but the worker-axis comparison on FastAPI is degenerate and is treated as a known limitation (§4).

FastAPI w2t2@1000 finally moves the needle: 152.7% CPU, approaching the 200% (2-core) cap, with P99 7.9 ms — the only FastAPI cell that's actually CPU-bound in the matrix.

# Finding 4 — Useful throughput per CPU% confirms the picture

![Successful RPS per 1% CPU — a normalised efficiency metric. Higher is more throughput per unit work.](3-bench/results/figures/cpu_efficiency.png){ width=100% }

FastAPI extracts ~6–15 successful requests/second per 1% CPU; BentoML extracts ~1–4 in non-collapsed cells and effectively 0 at collapse. Restated: BentoML's *overhead per useful request* is 3–10× higher than FastAPI's for this workload. This is the same conclusion in a different unit — the dispatcher is paying CPU for queue and batching machinery that doesn't translate into served requests at the model's 14 µs inference cost.

# Finding 5 — CPU rises during the attack window, not as a step

![Per-cell CPU% over the 45 s attack window for the BentoML collapse cell and FastAPI reference. BentoML's curve climbs as the queue depth grows; FastAPI is flat.](3-bench/results/figures/cpu_timeseries_collapse.png){ width=100% }

BentoML's CPU rises gradually over the first 10–15 s of the attack as queue depth and futures accumulate. FastAPI is flat from second one — no queueing means no build-up. The shape difference is itself a fingerprint: queueing systems show CPU growth tracking work-in-progress, immediate-dispatch systems do not.

# Worker-axis caveat (FastAPI)

Confirmed empirically here: FastAPI's w2* cells are dormant on CPU compared to w1* cells at matching threads, because Vegeta's keep-alive connection pool — small for sub-millisecond P50 — pins to one uvicorn worker. This means the headline Phase 3 FastAPI worker-axis comparison is degenerate. It does **not** invalidate the FastAPI-vs-BentoML headline because the relevant comparison is per-cell, and the per-cell numbers are correct as reported.

Planned follow-up: a small re-run with `vegeta attack -keepalive=false` against FastAPI w2* cells will force per-request TCP setup so the kernel's `accept()` spreads load across both workers. Expected outcome: w2 CPU ≈ 2× w1 CPU, w2 P99 improves slightly. ~15 min of bench time; documented as future work for this thesis.

# Conclusion

Direct CPU evidence confirms the dispatcher mechanism. At identical (1-worker, 1-thread, 600 RPS) load, BentoML burns ~2.5× the CPU as FastAPI and still collapses; the CPU clusters at ~100% (one core saturated) on 1-worker cells and ~200% on 2-worker cells, fingerprinting one async coroutine per worker. FastAPI's useful throughput per CPU% is 3–10× higher for this workload. For 0.014 ms tabular inference, adaptive batching's per-request plumbing cost exceeds any batching amortisation — the dispatcher is paying CPU for machinery that produces no benefit. Combined with the latency-distribution data from the headline report, the mechanism story is now quantitative on both sides: latency *pattern* (collapse RPS scales with workers) plus CPU *level* (one core per worker) plus CPU *shape* (build-up tracks queue depth).

# Artifacts

- Data: `project/3-bench/results/cpu_mechanism.csv` (60 CPU-instrumented runs across 20 cells).
- Raw timeseries: `project/3-bench/results/layer3_with_cpu/**/cpu_pct.csv`.
- Figures embedded above (`results/figures/cpu_*.png`).
- Plan + caveats: `thesis/phase3_cpu_mechanism_plan.md`.
