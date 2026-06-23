# 6. Results

## 6.1 L1 — inference floor

<<SOURCE: project/3-bench/results/layer1/L1_summary.json.>>

| Batch | per-call P50 | per-row P50 |
| ----: | -----------: | ----------: |
| 1 | **0.014 ms** | 0.014 ms |
| 16 | 0.349 ms | 0.022 ms |

<<TODO: 1 paragraph. The model itself costs ~14 µs per single-row
inference and is therefore ~1 % of any served request's total latency.
Everything else the report measures is *serving-stack overhead*. Forward-
reference Chapter 7's discussion of literature corroboration (`l1_literature_check.md`).>>

## 6.2 L0 — noise floor

<<SOURCE: project/3-bench/results/layer0/L0_summary.json.>>

| Endpoint | P50 | P99 |
| --- | ---: | ---: |
| FastAPI `/healthz` | 1.18 ms | 4.63 ms |
| BentoML `/livez` | 1.60 ms | 5.23 ms |

<<TODO: 1 paragraph: the floor is measured at 1 RPS over 60 s — each
request pays a cold connection setup cost because Vegeta's keep-alive
amortisation doesn't apply at that frequency. Steady-state (sustained
load) L2 P50 numbers are *below* the noise floor (FastAPI L2@150 RPS:
0.83 ms P50) because of this amortisation. Report this explicitly to
forestall the "your floor is higher than your data" question.>>

## 6.3 L2 — host HTTP

<<SOURCE: project/3-bench/results/layer2/<variant>/summary.json,
project/3-bench/results/budget.csv.>>

At (1 worker, 2 threads), 150 RPS, 45 s, server pinned to cores 0–1 via
`taskset`:

| Variant | L2 P50 (ms) | Framework overhead (L2 − L1, ms) |
| --- | ---: | ---: |
| `fastapi` | 0.609 | 0.595 |
| `bentoml_lat50` | 1.645 | 1.631 |
| `bentoml_lat250` | 1.529 | 1.515 |
| `bentoml_lat5` | 6.718 | 6.704 |

<<TODO: 1 paragraph. FastAPI's framework cost is ~0.6 ms. BentoML's
lat50/lat250 variants are ~1.5–1.6 ms — three times FastAPI. The lat5
variant is 6.7 ms even at 150 RPS in single-cell isolation because the
deadline is below the steady-state floor and the dispatcher is already
shedding load. This is the first quantitative evidence for the
configuration-artifact interpretation of the original calibration.>>

## 6.4 L3 — the matrix sweep

### 6.4.1 Headline metric: goodput within a 50 ms SLA

<<FIG: results/figures/goodput_50ms_vs_rps.png>>

<<TODO: 2 paragraphs. (a) The shape of the FastAPI curve: 99.7–100 %
across every parallelism cell from 50 to 1000 RPS. (b) The shape of the
BentoML curves: indistinguishable up to ~150 RPS, then a cliff. The cliff
RPS scales with worker count (one-worker cells collapse around 600 RPS;
two-worker cells hold to ~1000 RPS) and is largely independent of thread
count. This is the fingerprint of a per-worker bottleneck — the
dispatcher coroutine.>>

### 6.4.2 P99 inflection (the proposal's headline)

<<FIG: results/figures/inflection.png>>

<<TODO: 1 paragraph. FastAPI's P99 is essentially flat near 2–3 ms across
the matrix; BentoML's P99 climbs by two orders of magnitude. At the (1, 2)
cell, P99 progression for BentoML is 5 → 70 → 200 → > 3000 ms across
50/150/350/600 RPS.>>

### 6.4.3 Jitter (P99 / P50)

<<FIG: results/figures/jitter.png>>

<<TODO: 1 paragraph. FastAPI's jitter ratio stays in the 2–4× band even
at 1000 RPS. BentoML reaches 30× at 350 RPS and crosses 100× at
saturation. Adaptive batching does not compress jitter for this workload —
the dispatcher queue introduces variable per-request wait.>>

### 6.4.4 Tail distribution (CCDF)

<<FIG: results/figures/ccdf_tail.png>>

<<TODO: 1 paragraph. CCDF (P(latency > t)) at the (1, 2) cell, 600 RPS:
FastAPI's tail drops rapidly past 2 ms; BentoML's tail spans three
decades. The CCDF view reveals tail shape that percentile tables hide
(e.g., bimodality, heavy tails).>>

### 6.4.5 Latency budget

<<FIG: results/figures/budget_stacked_bar.png>>

<<TODO: 1 paragraph. Grouped bars per variant: L1 (~14 µs, identical
across variants), L2 (FastAPI 0.6 ms vs BentoML lat50/lat250 1.5–1.6 ms),
L3 (full deployment, +0.2–0.6 ms Docker overhead). Almost all measured
latency is serving-stack overhead, not inference. The lat5 budget reads
oddly (L2 > L3, "negative" Docker overhead) because the host-pinned
dispatcher at 5 ms deadline behaves differently from the container-pinned
one; cite the §4.8.2 train-time transient.>>

## 6.5 CPU mechanism evidence

<<SOURCE: project/3-bench/results/cpu_mechanism.csv, all four CPU
figures.>>

### 6.5.1 Same RPS, same parallelism, very different CPU

<<FIG: results/figures/cpu_bar.png>>

<<TODO: 1 paragraph quoting the headline cell: at (1 worker, 1 thread,
600 RPS), FastAPI uses ~38 % CPU while BentoML lat50 uses ~96 % and
collapses. Same model, same threads, same offered load — BentoML burns
~2.5× the CPU and still fails. The CPU is going into dispatcher plumbing,
not inference. Pull exact numbers from `cpu_mechanism_agg.csv` once the
overnight run finishes.>>

### 6.5.2 BentoML's CPU saturates one core per worker

<<TODO: 1 paragraph. On the (1, *) BentoML cells, mean CPU clusters
around 95–100 % — close to *one core*, not two. That is the signature
of a single async coroutine pinned to one core. The (2, *) cells run at
roughly twice the CPU, matching the doubled dispatcher count, which is
why the collapse RPS approximately doubles with worker count.>>

### 6.5.3 FastAPI tracks the thread axis, not the worker axis

<<FIG: results/figures/cpu_vs_p99.png>>

<<TODO: 1 paragraph. FastAPI at w1t1@600 (~38 %) and w2t1@600 (~38 %)
are statistically indistinguishable; w1t2@600 (~120 %) is roughly 3× the
w1t1 value, matching ORT's second intra-op thread becoming active. The
worker axis is degenerate under HTTP/1.1 keep-alive (the connection pool
pins to a single uvicorn worker). This is reported as a §4.8.1 caveat;
the FastAPI-vs-BentoML headline comparison is unaffected because it is
per-cell.>>

### 6.5.4 Useful throughput per CPU%

<<FIG: results/figures/cpu_efficiency.png>>

<<TODO: 1 paragraph. FastAPI extracts ~6–15 successful requests/sec per
1 % CPU; BentoML extracts ~1–4 in non-collapsed cells and effectively 0
at collapse. Restated: BentoML's *overhead per useful request* is 3–10×
higher than FastAPI's for this workload.>>

### 6.5.5 CPU rises during the attack window

<<FIG: results/figures/cpu_timeseries_collapse.png>>

<<TODO: 1 paragraph. BentoML's CPU rises gradually over the first
10–15 s of the attack as queue depth and futures accumulate; FastAPI is
flat from the first sample. The shape difference is itself a fingerprint:
queueing systems track work-in-progress; immediate-dispatch systems do not.>>

## 6.6 Statistical confidence

<<SOURCE: project/3-bench/results/ci.csv, effect_sizes.csv.>>

### 6.6.1 Bootstrap CIs on percentiles

<<TODO: 1 paragraph. CI widths at n=3 are wide in absolute terms (factor
of 2–3 on P99 in low-error cells); we report them honestly and rely on
the size of the FastAPI–BentoML gap (orders of magnitude in P99) for
inferential weight. Cite the Efron 1979 bootstrap-percentile method.>>

### 6.6.2 Cliff's δ across variants

<<TODO: 1 paragraph. Cliff's δ is in the ±1 range — large effect — for
FastAPI vs every BentoML variant at every load. The non-overlap is
robust against scale; the comparison is meaningful at small n.>>

### 6.6.3 Repeat-to-repeat stability (CoV)

<<TODO: 1 paragraph. CoV across the 3 repeats per cell is typically
<10 % for FastAPI and <20 % for BentoML below collapse. Above collapse
BentoML's CoV jumps to >50 % — variance itself is a saturation
fingerprint.>>

## 6.7 Summary of findings

<<TODO: 1 paragraph summarising before the discussion: FastAPI dominates
the matrix at this scale; the dispatcher is the bottleneck; the
deadline-knob is a configuration-vs-architecture distinction; the L1
floor is plausibly correct; the worker-axis caveat is real but bounded.>>
