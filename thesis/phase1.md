# Phase 1 — Model Development & ONNX Export

## Purpose

Phase 1 produces a single, version-pinned ONNX artifact that both serving pipelines load. The artifact is the *contract* between Phase 1 and Phase 2: both servers verify the file's SHA256 at startup against the value committed in `schema.json`, so any latency difference observed in Phase 3 is attributable to the serving stack, not to model drift between runs.

## Dataset — HIGGS

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Rows | 11,000,000 |
| Features | 28 numeric (low- and high-level kinematic variables) |
| Label | binary (signal vs background) |
| Storage | gzipped CSV ~7.5 GB; converted to Parquet for fast reload |
| Train/test split | first 10,500,000 / last 500,000 (canonical HIGGS split, comparable to the literature) |

HIGGS was chosen because it is purely numeric, dense, and large. This eliminates preprocessing variability as a confounder: every request the server receives is a flat 28-element float32 array, so request-side measurement reflects model execution only.

## Model — XGBoost binary classifier

| Hyperparameter | Value |
|---|---|
| Algorithm | XGBoost 2.1.4, `hist` tree method |
| Trees | 500 |
| Depth | 8 |
| Learning rate | 0.1 |
| Seed | 42 |
| Train time | 250.6 s on the project's reference host |

Resulting metrics on the 500k-row test split: **AUC 0.8406, accuracy 0.757**.

These numbers are *not* the thesis's deliverable — the thesis is about serving-stack latency, not predictive accuracy. The model exists only to give the servers a non-trivial workload (500 trees of depth 8 is enough that single-request inference takes a measurable ~2–5 ms, but not so deep that ONNX export becomes the bottleneck).

**Frozen hyperparameters.** Tree count, depth, and seed are explicitly locked for the duration of the experiment. Bumping any of them invalidates every Phase 3 latency measurement collected so far. The plan explicitly leaves "tree count" as the one knob to reach for if Phase 3 saturates both pipelines identically and the differences are unmeasurable; in that case the trees increase to 1000 and the entire matrix is rerun.

## Export — ONNX opset 15

The trained booster is exported to ONNX via `onnxmltools.convert_xgboost` with `target_opset=15` and input tensor `("features", FloatTensorType([None, 28]))`. The export produces:

- `model.onnx` — 8.3 MB, single input named `features`, two outputs named `label` and `probabilities`.
- `schema.json` — input dtype/shape, output names, the SHA256 of `model.onnx`, and the parity report.

### Parity check

After export, the script loads both predictors and scores 10,000 random rows from the test set through each:

- XGBoost Python API: `model.predict_proba(X)[:, 1]`
- ONNX Runtime: `sess.run(None, {"features": X})[1][:, 1]`

It asserts `max(|py_proba - onnx_proba|) < 1e-5`. Measured: **max abs diff = 4.17e-7, mean abs diff = 6.08e-8**.

This number — three orders of magnitude tighter than the tolerance — gives downstream Phase 2 parity checks plenty of headroom. Phase 2's server-vs-local-ONNX parity comes in at 2.98e-7, also well under tolerance.

## ONNX export gotchas (learned the hard way)

These are the non-obvious issues that bit Phase 1 development. Documenting them here so that any future regenerate-the-model step doesn't have to rediscover them:

1. **`FloatTensorType` is shipped by three different packages** (`onnxconverter_common`, `skl2onnx.common.data_types`, `onnxmltools.convert.common.data_types`). They are *not* interchangeable — the onnxmltools shape calculator rejects the other two via `isinstance`. Always import from `onnxmltools.convert.common.data_types`.
2. **`onnxmltools.convert_xgboost` does not accept `options=`**. Trying to route through `skl2onnx.convert_sklearn` to get `zipmap=False` requires the skl2onnx `FloatTensorType`, which then trips the type check above. With the currently pinned `onnxmltools`, the XGBoost converter does **not** emit a trailing `ZipMap` — `probabilities` is already a clean float32 `[N, 2]` ndarray (verified in Netron and asserted in the parity check). Both Phase 2 servers can read `outputs[1][:, 1]` directly; no list-of-dict handling is needed.
3. **`xgboost==2.1.x` is incompatible with `scikit-learn>=1.7`** — sklearn 1.7 changed the `_estimator_type` mechanism and XGBoost's sklearn wrapper still references it as an attribute. `project/1-model/pyproject.toml` pins `scikit-learn<1.7` as the workaround.

## Phase 1 → Phase 2 handoff

The handoff is **two files** in `project/1-model/artifacts/`:

- `model.onnx` — the model, opset 15, input `features` float32 `[None, 28]`
- `schema.json` — the descriptor, including `model_sha256` (here `dcc3b14573…`) and the parity report

Both Phase 2 servers, at startup, hash `model.onnx` and assert the hash matches `schema.json["model_sha256"]`. Mismatch causes the process to fail immediately rather than silently serve a stale or swapped model. The bench harness also captures this SHA from `/version` into each result directory so post-hoc analysis can prove which model produced which numbers.

## Reproducing Phase 1

```bash
make setup-model    # uv sync the Phase 1 venv
make download       # download HIGGS (~7.5 GB, one-time)
make train          # train XGBoost (~4 min)
make export         # convert to ONNX + run parity check
make verify         # re-run parity check only
```

For development without committing to the 7.5 GB download:

```bash
make smoke          # train+export on 50k synthetic rows, ~30 s
```

## Phase 1 exit criteria (all met)

- `model.onnx` exists in `project/1-model/artifacts/`.
- `schema.json` exists with `model_sha256`, parity report, opset 15.
- XGBoost↔ONNX parity check passes with max abs diff < 1e-5.
- `model_card.md` documents the model's predictive metrics for the thesis's "model description" section.
