# Phase 3 follow-up — CPU mechanism overnight run (plan + caveats)

**Scheduled:** 2026-06-15 02:00, Windows Task Scheduler entry
`TG-BenchCpuMechanism`. WakeToRun + StartWhenAvailable + battery guards off.
ExecutionTimeLimit 72 h (run itself is ~90 min).

**Purpose.** Add server-side CPU evidence to the Phase 3 mechanism claim that
"BentoML's per-worker dispatcher coroutine serialises requests, not lack of
compute". The headline L3 sweep captured only Vegeta-side latency; without
CPU% the *mechanism* is inferred but not measured. This follow-up measures it
directly.

---

## 1. What runs

`project/3-bench/layers/L3_cpu_mechanism.py` — 20 hand-picked cells × 3 repeats
× 45 s attack, with `docker stats --no-stream` polled at ~1 Hz against the
running server container. Output under
`project/3-bench/results/layer3_with_cpu/<variant>/w<W>_t<T>/rps<RPS>/run<N>/`.

### Cell selection (20)

| Variant | Cells | Why |
| --- | ---: | --- |
| `fastapi` | 5 | Reference: w1t1@600, w1t2@600, w2t1@600, w2t2@600, w2t2@1000. Spans both axes at saturation, includes the only cell that's truly CPU-bound (w2t2@1000). |
| `bentoml_lat50` | 8 | The headline variant: w1t1@350, w1t1@600, w1t2@350, w1t2@600, w2t1@600, w2t1@1000, w2t2@600, w2t2@1000. Brackets the collapse RPS on both worker counts so the dispatcher-coroutine claim is testable. |
| `bentoml_lat250` | 4 | Looser deadline, real batches form: w1t2@350, w1t2@600, w2t2@600, w2t2@1000. Lets the report say "even with room to batch, plumbing dominates." |
| `bentoml_lat5` | 3 | Calibration-artifact reference: w1t2@150, w1t2@350, w2t2@600. Tight deadline, mostly to show CPU pattern is similar at lower load. |

**Total**: 5 + 8 + 4 + 3 = 20 cells × 3 repeats = 60 runs. At ~85 s per cell
wall-clock (server boot + warmup + 45 s attack + teardown + settle), expected
wall time ~85 min.

### Pipeline

`run_cpu_mechanism.ps1` (registered task):

1. Verify `docker info` responds; abort if not.
2. Clear any prior `results/layer3_with_cpu/` tree (idempotent fresh start).
3. Run the bench under WSL2 Ubuntu via
   `uv run python layers/L3_cpu_mechanism.py --duration 45 --repeats 3`.
4. Run `analyze.py --no-encode` to refresh `summary.csv`, `ci.csv`,
   `effect_sizes.csv`, and (new) `cpu_mechanism.csv`.
5. Run `cpu_mechanism_report.py` to emit the plot suite + final MD report.

All steps tee to `results/layer3_with_cpu_logs/run_<stamp>.log`.

---

## 2. Caveats (already known — read before interpreting morning results)

### 2.1 FastAPI worker-axis dormancy (high relevance)

Empirically confirmed in the 7-cell smoke: FastAPI `w2t1@600` and `w1t1@600`
have **identical** mean CPU (38%). Same for `w2t2` vs `w1t2`. The cause is
HTTP/1.1 keep-alive + Vegeta's small connection pool — at sustained 600 RPS
with P50 < 1 ms, one TCP connection saturates one worker's accept queue, and
the second worker never receives a connection.

**Implication for this run**:
- FastAPI `w2*` cells will look identical to `w1*` cells on CPU, throughput,
  and latency. This is *not a bug in the harness* and not a new finding to
  re-verify — it's reproducing the known artifact.
- The mechanism comparison (FastAPI vs BentoML at the **same (1,*)** cell) is
  still valid because the dormancy affects only the worker axis on FastAPI.
- The morning report should *cite the dormancy* and refer to the planned
  follow-up (§6 below).

### 2.2 Dispatcher train-time transient (medium relevance)

BentoML's dispatcher has a 30-second `train_optimizer` warm-up phase during
which the dispatcher's batch-size estimator stabilises. The original L3 sweep
used 30-request warmup; we bumped to 100 to drain this phase. This run uses
WARMUP=100. If a run shows high error rate in `run1` only and clean rates in
`run2`/`run3`, that's the same transient — pool the 2-3 repeats not the 1-3.

### 2.3 Sampler density vs steady-state (medium)

`docker stats --no-stream` blocks ~2 s per call on WSL2, so ~22 samples per
45-s cell. Sufficient for mean and P95, marginal for instantaneous detection of
short bursts. The sampler also typically catches one teardown-window frame at
the tail (a sub-1% sample); filter `cpu_pct < 5.0` from the timeseries when
plotting steady-state, but include all samples in the mean (the teardown
sample's weight is negligible).

### 2.4 Headline dataset preservation (critical)

The original 219-cell dataset is **physically separate** — snapshotted to
`results/layer3_no_cpu_snapshot/` (941 MB, one-time) and the live tree at
`results/layer3/` is left untouched. The overnight run writes to
`results/layer3_with_cpu/` only. **No headline number can be invalidated by
this re-run**, even if the bench crashes mid-way.

### 2.5 Repeat counts vs original sweep (low)

The CPU subset uses 3 repeats matching the headline. CI widths are computed
from those 3 repeats — narrow CIs would be unusual at n=3, so the report
should describe them as "consistent with" rather than "tightly confirming."
The CIs on CPU% are more informative than on P99 here.

### 2.6 Loopback noise floor caveat (low)

The L0 noise floor (FastAPI `/healthz` at 1 RPS) measured 1.18 ms P50 — higher
than the FastAPI L3 P50 at 150 RPS (0.83 ms). Reason: at 1 RPS each request
pays cold connection setup; at sustained load Vegeta's keep-alive amortises
that cost. The noise floor is the **upper bound on per-request transport
overhead at low load**, not a floor on steady-state latency. Report should
state this explicitly to avoid the "your floor is higher than your data"
objection.

---

## 3. Success criteria (what makes morning a green light)

- `results/layer3_with_cpu/` contains 60 cell directories with both
  `summary.json` and `cpu_pct.csv` present, all `cpu_n_samples > 15`.
- `results/cpu_mechanism.csv` exists with the 60 rows.
- The post-bench script wrote `phase3_cpu_mechanism_report.md` + 4 PNGs under
  `results/figures/`.
- The headline `results/summary.csv` row count is still 219 (the original
  sweep), proving the overnight run didn't touch it.

If any of these fail, the snapshot at `results/layer3_no_cpu_snapshot/` is
the recovery point.

---

## 4. Expected findings (priors before the run)

| Claim | Expected morning evidence |
| --- | --- |
| BentoML burns more CPU than FastAPI at the same RPS | `bentoml_lat50` w1t1@600 mean CPU > 90%; `fastapi` w1t1@600 mean CPU < 50%. **Smoke already showed 96% vs 38%** — the run quantifies CIs. |
| BentoML's collapse is dispatcher-bound, not CPU-bound | At collapse, BentoML mean CPU ~ 95-100% on a **single core** (≈ one asyncio coroutine pinned). Doesn't grow with more workers per cell — w2t* cells see CPU split across the two dispatcher coroutines. |
| FastAPI CPU scales with thread axis, not worker axis | w1t1 ≈ w2t1 on CPU; w1t2 ≈ w2t2 on CPU; w?t2 ≈ 2× w?t1. Smoke confirmed this. |
| `lat250` BentoML still doesn't beat FastAPI | CPU still higher than FastAPI at every RPS; goodput maybe better than `lat50` but not crossing FastAPI's line. |

Surprising outcomes (would prompt extra investigation):

- BentoML mean CPU < 60% at collapse → dispatcher is doing *less* work than
  expected; the bottleneck might be a synchronisation primitive, not the
  event loop itself.
- FastAPI mean CPU > 60% at w1t1@600 → smoke was anomalous; the 38% figure
  was an outlier.
- BentoML w2 CPU not double w1 CPU → suggests the "per-worker dispatcher"
  story is wrong somehow; would force a re-read of the dispatcher source.

---

## 5. Deliverables in the morning

Located at `project/3-bench/results/`:

- `cpu_mechanism.csv` — 60 rows: per-cell percentiles + CPU mean/p50/p95/max.
- `figures/cpu_bar.png` — grouped bar chart, mean CPU per cell with 95% CIs.
- `figures/cpu_vs_p99.png` — scatter of mean CPU vs P99, one point per cell.
- `figures/cpu_efficiency.png` — requests/sec per CPU% (a normalised
  "throughput per unit work" metric).
- `figures/cpu_timeseries_collapse.png` — CPU over time during the BentoML
  collapse cell, showing dispatcher saturation onset.
- `phase3_cpu_mechanism_report.md` — pandoc-ready report, ~3 pages, sections:
  Context, Method, Findings (one per plot), Worker-axis caveat, Conclusion.

The wrapper does **not** generate a PDF — that's a single `_md2pdf.sh` call
the next morning, kept manual so any morning edits to the MD are picked up.

---

## 6. Planned but not part of this run

**FastAPI no-keep-alive follow-up.** A small script
`layers/L3_fastapi_no_keepalive.py` (not yet written) would re-run FastAPI
(1,2), (2,1), (2,2) at 350/600/1000 RPS with `vegeta attack -keepalive=false`,
forcing per-request TCP setup so the kernel `accept()` spreads load across
both uvicorn workers. ~15 min of bench time. Output goes to
`results/layer3_no_keepalive/`. This is the proof that worker-axis dormancy is
client-driven, not server-driven, and would close the §"Limitations" gap in
the headline report. Run after the morning report is reviewed.

---

## 7. Recovery / abort

- **Cancel the run**: `schtasks /Delete /TN TG-BenchCpuMechanism /F`.
- **Re-arm for tomorrow**:
  ```powershell
  schtasks /Create /TN "TG-BenchCpuMechanism" `
    /TR "powershell.exe -ExecutionPolicy Bypass -NoProfile -File C:\Users\David\Documents\learning_repos\TG\project\3-bench\run_cpu_mechanism.ps1" `
    /SC ONCE /ST 02:00 /SD (next date) /F
  ```
- **Inspect logs**: `results/layer3_with_cpu_logs/run_<stamp>.log` is the full
  bench stdout for that night.
- **Roll back to snapshot**: if the headline tree somehow gets corrupted
  (shouldn't be possible — the runner writes only to `layer3_with_cpu/`),
  `cp -r results/layer3_no_cpu_snapshot/. results/layer3/` restores it.
