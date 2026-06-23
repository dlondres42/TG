# TG — P99 Tail Latency for Tabular ML Serving

Graduation thesis (CIn/UFPE) comparing **FastAPI + ONNX Runtime** vs **BentoML +
adaptive batching** as serving stacks for an XGBoost model on tabular data.
The deliverable is a quantitative analysis of latency distributions (P50/P95/P99)
under varying concurrency. Full plan: [PLAN.md](PLAN.md).

## Repo layout

```
pre_projeto/      proposal LaTeX + compiled PDF (frozen — don't edit)
thesis/           final monograph LaTeX + per-phase write-ups
project/          implementation, numbered by thesis phase:
  1-model/        Phase 1 — train XGBoost on HIGGS, export to ONNX
  2-serving/      Phase 2 — two containerised inference servers:
    fastapi/      FastAPI + ONNX Runtime container
    bentoml/      BentoML + adaptive batching container
  3-bench/        Phase 3 — Vegeta-driven benchmark harness + analysis
  phase2_serving_report.md   one-page Phase 2 checkpoint report
PLAN.md           high-level execution plan (ground truth for scope)
Makefile          orchestrates env setup, builds, and phase commands
```

## Environment strategy

**One uv venv per phase**, not one shared conda env. Each `project/<phase>/` has
its own `pyproject.toml` and lockfile. Reasoning: BentoML pulls a heavy dep
tree (pydantic, prometheus, opentelemetry) that fights with the model-training
stack (`xgboost`, `onnxmltools`, `scikit-learn`); per-phase venvs prevent those
conflicts and keep each Dockerfile reproducible from its own `pyproject.toml`.

- **Python:** 3.11 (pinned in `.python-version`; xgboost/onnxruntime/bentoml
  all ship clean Windows wheels for it).
- **Package manager:** `uv` — fast lockfile-based resolution.
- **Locking:** `uv.lock` is committed per phase for thesis reproducibility.

## Common commands

```bash
make help              # list all targets
make setup             # uv sync all 4 phase venvs
make setup-model       # just Phase 1
make smoke             # synthetic-data train+export end-to-end (~30s, no download)
make download          # fetch HIGGS (~7.5 GB, one-time)
make train             # train XGBoost on HIGGS
make export            # export to ONNX + XGBoost<->ONNX parity check
make verify            # rerun parity check only
```

`make smoke` is the fast path for validating that the model phase works without
committing to the 7.5 GB HIGGS download.

## Phase 1 artifact contract

The Phase 1 → Phase 2 handoff is a single ONNX file plus a schema descriptor:

- `project/1-model/artifacts/model.onnx` — opset 15, input `features` float32 `[None, 28]`
- `project/1-model/artifacts/schema.json` — input/output names, SHA256 of the ONNX
  file, and the XGBoost↔ONNX parity report (max-abs prob diff < 1e-5)

Both servers in Phase 2 load `model.onnx` and verify the SHA256 against
`schema.json`, so any latency difference between FastAPI and BentoML is
attributable to the serving stack, not the model.

## ONNX export gotchas (learned the hard way)

- **`FloatTensorType` is shipped by three different packages** (`onnxconverter_common`,
  `skl2onnx.common.data_types`, `onnxmltools.convert.common.data_types`). They are not
  interchangeable — the onnxmltools shape calculator rejects the other two via
  `isinstance`. Always import from `onnxmltools.convert.common.data_types`.
- **`onnxmltools.convert_xgboost` does not accept `options=`.** Routing through
  `skl2onnx.convert_sklearn` to get `zipmap=False` requires the skl2onnx
  `FloatTensorType`, which then trips the onnxmltools shape-calculator type check
  above. With the currently pinned `onnxmltools` the XGBoost converter does **not**
  emit a trailing `ZipMap` — `probabilities` is a clean float32 `[N, 2]` ndarray
  (verified in Netron and asserted in the parity check). Phase 2 servers can read
  `outputs[1][:, 1]` directly; no list-of-dict handling needed.
- **`xgboost==2.1.x` is incompatible with `scikit-learn>=1.7`** — sklearn 1.7 changed
  the `_estimator_type` mechanism and XGBoost's sklearn wrapper still references it
  as an attribute. Pinned `scikit-learn<1.7` in `project/1-model/pyproject.toml`.

## Constraints worth remembering

- **Don't change Phase 1 hyperparameters mid-experiment** — if `n_trees`,
  `max_depth`, or seed change, the latency comparison resets and old bench
  results become incomparable. Bumping tree count is the one knob the plan
  explicitly leaves open if both pipelines saturate identically.
- **Latency budget:** the proposal commits to comparing single-host runs with
  CPU pinning (server cores 0–1, Vegeta cores 2–3) and 2 CPU / 2 GiB container
  limits. Both Dockerfiles and the bench harness must respect this.
- **Vegeta is open-loop:** any benchmark result that uses `wrk`, `ab`, or
  closed-loop loaders is invalid per the proposal — Coordinated Omission
  must be ruled out (see proposal §3.3 and `tema1.txt`).
