# Phase 3 — Benchmark & Latency Analysis (Corrected)

## Purpose

Phase 3 produces the thesis's primary quantitative result: latency
distributions and tail behaviour for both Phase 2 pipelines under open-loop
load on an isolated single host. Beyond raw totals it **decomposes** latency
into stack layers (model / framework+HTTP / Docker) and reports a
**goodput-under-shared-SLA** metric that compares a load-shedding system
(BentoML) and a queueing system (FastAPI) on equal footing.

This document is the corrected version. The first measurement pass produced
a result so adverse to BentoML that we re-examined the methodology, found a
calibration artifact in the configuration of `max_latency_ms`, and re-ran
BentoML at three values of that knob to disentangle the artifact from the
underlying truth.

## Experimental controls

- Open-loop Vegeta load (proposal §3.3; Coordinated Omission ruled out).
- Single host, disjoint CPU pinning: server container cores 0–1, Vegeta
  sidecar container cores 2–3, both `--cpus=2 --memory=2g`. Cores 4–11 idle.
- Frozen model: every server verifies the Phase 1 SHA-256 at startup and logs
  it into each cell's `version.json`.
- Vegeta with `-timeout=3s` so a saturated server fails fast instead of
  hanging Windows ephemeral ports.

## The methodological correction

`max_latency_ms` is **not** a "batch wait window to minimise" — reading BentoML's
own dispatcher source ([dispatcher.py:303-311](project/2-serving/bentoml/.venv/Lib/site-packages/bentoml/_internal/marshal/dispatcher.py))
confirms it is a **request-completion deadline**:

```python
latency_0 = w0 + a * n + b                # est. latency to clear the queue now
if n > 1 and latency_0 >= self.max_latency:
    self._queue.popleft().future.cancel()  # → fallback() → HTTP 503 "overloaded"
```

A request is shed whenever its estimated total latency exceeds `max_latency`.
With `max_latency = 5 ms` and a measured serving-overhead floor of ~3–4 ms
(L2 − L1), there is only ~1 ms of headroom; any queue (inevitable at 150+ RPS)
breaches the deadline and triggers mass 503s.

Three compounding flaws in the original Phase 2 calibration:

1. **BentoML warned us** (dispatcher.py:278): it logs *"max latency likely too
   low for serving"* when `o_a + o_b >= max_latency`. The warning is present
   in our Phase 2 logs and was ignored.
2. **The calibration objective was wrong**: it swept `max_latency_ms ∈ {2,5,10}`
   at a single low load (100 RPS) and picked the lowest-P99/zero-error value.
   Because the knob is a deadline, *lower = more fragile under load* — the
   procedure structurally drove the knob toward the value most fragile at the
   Phase 3 RPS levels.
3. **Apples-to-oranges error semantics**: BentoML enforces its SLA by shedding
   load (503); FastAPI has no SLA and absorbs overload as latency. Comparing
   error rates directly was unfair.

## Corrected design

- BentoML is split into three **variants** by `max_latency_ms`:
  `bentoml_lat5` (original artifact, 39 cells preserved), `bentoml_lat50`
  (realistic web-tier SLA ~10× the serving floor, 51 cells), `bentoml_lat250`
  (a generous SLA where batching can fully form, 47 cells). `max_batch_size=16`
  fixed across all three.
- A new fair metric, **goodput@SLA**, counts a request as successful only if
  it returned 2xx within an SLA latency. We report SLA = 50 ms and 100 ms.
  This treats a BentoML 503 and a FastAPI 200-with-2-second-latency as
  equivalent SLA failures.
- FastAPI's 60-cell dataset from the first pass is retained unchanged.

## Harness architecture (`project/3-bench/`)

| Component | Role |
| --------- | ---- |
| `lib.py` | Server lifecycle (`run_server` with optional per-variant mounts/env), Vegeta sidecar (`vegeta_attack`), targets builder, `vegeta_latency_status_ns` (encodes raw `.bin` to per-request latency + HTTP code, powers P99.9 and goodput@SLA) |
| `targets/generate_targets.py` | Samples 1000 real HIGGS test rows into `bodies.jsonl`; the harness base64-encodes them into a Vegeta JSON targets file per pipeline |
| `layers/L1_inference_floor.py` | Pure ONNX microbenchmark (batch sizes 1/4/8/16) |
| `layers/L2_host_http.py` | Per-variant server on host via `uv run` + Vegeta sidecar (`--max-latency-ms-list`) |
| `layers/saturation_pilot.py` | Ramps RPS per cell to find each one's knee |
| `layers/L3_docker_sweep.py` | Per-variant matrix sweep — containerised, resumable, early-aborting. BentoML variants are passed via `--max-latency-ms-list`; each variant mounts a generated `batching.json` into the container at `/work/batching.json` and sets `BATCHING_PATH` |
| `analyze.py` | Aggregates results → `summary.csv` (with per-cell goodput@SLA cached into `summary.json`), `budget.csv`, figures |

## The three-layer decomposition

| Layer | Isolates | Result at (1w/2t), 150 RPS |
| ----- | -------- | -------------------------- |
| **L1** inference floor | model's intrinsic CPU cost (shared) | 0.046 ms P50 |
| **L2** host HTTP | framework + parsing + TCP loopback | fastapi 1.57 / lat5 3.38 / lat50 5.27 / lat250 9.27 |
| **L3** containerised | + Docker | fastapi 2.00 / lat5 5.44 / lat50 3.00 / lat250 6.56 |

The framework band grows monotonically with `max_latency_ms` (1.5 → 3.3 → 5.2
→ 9.2 ms) — a direct, visible cost of looser SLAs. The Docker layer
(L3 − L2) is clean for FastAPI (+0.43 ms) and the original lat5 (+2.05 ms) but
**negative** for lat50 and lat250 (−2.3 and −2.7 ms). This is itself a
finding: BentoML's dispatcher adapts to its environment — on the unpinned host
it estimates available throughput differently and tolerates longer waits, so
`L3 − L2` no longer subtracts cleanly. The budget chart is rigorous for
FastAPI's serial path and informative-but-not-additive for BentoML.

## Matrix results — goodput within 50 ms SLA

Median across repeats. `—` = early-aborted past collapse. The (1,2) cell is
the calibration cell; the (2,1) cell is BentoML's strongest parallelism cell
(process replication).

### Cell (1 worker, 2 threads)

| RPS  | fastapi | lat5 | lat50 | lat250 |
| ---: | ------: | ---: | ----: | -----: |
| 50   | 100 %   | 99.5 % | 99.7 % | 99.4 % |
| 150  | 100 %   | 95.7 % | 98.2 % | 96.8 % |
| 350  | 100 %   | **6.4 %** | **75.3 %** | 38.5 % |
| 600  | 100 %   | —    | 0.6 %  | 0.1 %  |
| 1000 | 99.5 %  | —    | —     | —      |

### Cell (2 workers, 1 thread) — BentoML's best parallelism

| RPS  | fastapi | lat5 | lat50  | lat250 |
| ---: | ------: | ---: | -----: | -----: |
| 350  | 100 %   | 97.5 % | 97.8 % | 97.3 % |
| 600  | 100 %   | 26 %   | **84 %** | 11 %   |
| 1000 | 99.9 %  | —    | 0.04 %  | 0.02 % |

## Interpretation

Three things fall out cleanly:

1. **The original `lat5` collapse was largely artifact.** At 350 RPS in the
   (1,2) cell, `lat5` reported 6 % goodput-within-50 ms; with the realistic
   `lat50` deadline the same hardware/model serves **75 %**. The "5–10×
   earlier collapse" headline was a measurement flaw, not a property of
   adaptive batching.

2. **Even after the correction, FastAPI dominates.** The honest truth is that
   inference at 0.05 ms is too cheap to amortise via batching — the dispatcher
   only ever adds overhead. FastAPI maintains 99.5–100 % goodput at 50–1000
   RPS across every parallelism cell; BentoML's best variant (`lat50`) trails
   at every RPS in every cell. The user's a-priori intuition that BentoML
   should win under concurrency is not borne out for *this* workload — but
   the reason is fundamental (cheap inference defeats batching), not
   configuration error.

3. **The `max_latency_ms` knob has a sweet spot that isn't either extreme.**
   `lat5` collapses prematurely via 503-shedding; `lat250` over-corrects
   (requests succeed but pay the deadline as latency and miss the SLA);
   `lat50` consistently wins among the three. The thesis-level lesson: batching
   frameworks must be benchmarked against the latency SLA they will operate
   under in production, not against synthetic low-load probes.

## Figures (`results/figures/`)

- `budget_stacked_bar.png` — per-variant L1 / (L2−L1) / (L3−L2). FastAPI is the
  reference; BentoML's framework band scales with `max_latency_ms`; the
  Docker band is unstable for BentoML (load-dependent dispatcher behaviour —
  see decomposition caveat above).
- `inflection.png` — P99 vs RPS, one line per variant per parallelism cell.
- `jitter.png` — P99/P50 ratio vs RPS.
- **`goodput_50ms_vs_rps.png`** (the fairness-corrected headline plot) —
  goodput within 50 ms SLA per variant per cell. Treats BentoML 503s and
  FastAPI's slow-200s as equivalent SLA misses.
- `goodput_100ms_vs_rps.png` — same metric at a 100 ms SLA.

## Gotchas (learned during the sweeps)

1. **`max_latency_ms` is a deadline, not a wait window.** The calibration that
   minimises low-load P99 produces the most fragile value for high concurrency.
   Always calibrate against goodput under the production SLA.
2. **Windows ephemeral-port exhaustion** (`WinError 10048`): a saturated
   BentoML server held connections open for Vegeta's 30 s default; the pileup
   exhausted ports and crashed the first sweep. Fixed with Vegeta `-timeout=3s`
   + per-run `try/except` + settle delay between cells.
3. **Early-abort**: once a cell exceeds 50 % errors at a rate, higher rates
   are not swept — they only add 100 %-error, port-exhausting cells.
4. **Budget alignment**: L2 and L3 must be compared at the *same* RPS; the
   budget uses 150 RPS (lowest non-trivial point in the shared grid). The
   alignment also requires the variant's L2 measurement, so L2 was re-run
   for `lat50` and `lat250`.
5. **BentoML budget decomposition caveat**: `L3 − L2` is meaningful for
   FastAPI but not for BentoML at sane SLAs — the dispatcher adapts to
   available CPU, so the host-vs-container difference isn't pure "Docker
   overhead." The framework band itself remains informative.
6. **BentoML dispatcher warmup transient.** The first cell-run after a
   container boot can record a burst of vegeta code-0 responses (no HTTP
   reply) that vanish on subsequent runs of the same cell. The cause is
   BentoML's `train_optimizer` phase (see [dispatcher.py:198-284](project/2-serving/bentoml/.venv/Lib/site-packages/bentoml/_internal/marshal/dispatcher.py))
   — the dispatcher samples several batch sizes to learn its `(o_a, o_b)`
   per-request-cost estimates; futures pending during training can be
   cancelled in ways that drop the HTTP connection. The warmup count must be
   large enough for `train_optimizer` to complete before measurement. We
   bumped the per-cell warmup from 30 to **100 requests** (`WARMUP=100` in
   `layers/L3_docker_sweep.py`); at warmup=30 the artefact concentrated in
   `bentoml_lat250` at 50 RPS (46–70 % code-0 on first-runs only, with the
   same cell's run2/run3 clean at 100 % 200). After the bump, all 50-RPS
   cells across all variants returned 100 % 200s.
6. **Per-worker batching** (from Phase 2): BentoML's 2-worker cells split load
   across two independent dispatchers; the `lat=N` value applies per dispatcher.

## Reproduce

```bash
# Phase 2 images and Phase 1 artifacts must exist
make build-servers
make gen-targets
make bench-l1                                       # inference floor

# Layer 2 — host HTTP, per variant
cd project/3-bench && uv run python layers/L2_host_http.py \
    --pipelines fastapi bentoml --max-latency-ms-list 5 50 250 \
    --rps 150 --duration 45

# Layer 3 — containerised matrix (the long run; resumable)
uv run python layers/L3_docker_sweep.py \
    --pipelines fastapi bentoml --max-latency-ms-list 5 50 250

# Analysis (encodes bins for P99.9 and goodput@SLA, caches into summary.json)
uv run python analyze.py --sla-ms 50 100
```

## Phase 3 exit criteria (met)

- L1 floor measured; L2 measured for `fastapi`, `bentoml_lat5`, `bentoml_lat50`,
  `bentoml_lat250` at (1,2)@150 RPS.
- L3 sweep complete: 197 cells across 4 variants (60 fastapi + 39 lat5 + 51
  lat50 + 47 lat250); each cell has 3 repeats up to its collapse rate.
- `summary.csv` populated with per-cell P50/P95/P99/P99.9, error rate, and
  goodput@{50,100}ms; `budget.csv` per variant.
- Five figure types produced under `results/figures/`.
- This document and `project/phase3_bench_report.md` rewritten with the
  corrected findings; `thesis/phase2.md` calibration sections updated.
