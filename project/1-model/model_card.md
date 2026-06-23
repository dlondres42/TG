# Model Card — TG Phase 1

## Overview

XGBoost binary classifier trained on the HIGGS dataset. The model is **not the
research subject** — it is a fixed serving target whose latency under different
serving stacks is what the thesis measures.

## Intended use

- **In-scope:** latency benchmarking of inference serving pipelines (FastAPI vs BentoML).
- **Out-of-scope:** any actual high-energy physics inference. AUC/accuracy figures are
  reported for honesty, not because they're the deliverable.

## Data

- **Source:** UCI ML Repository — HIGGS (https://archive.ics.uci.edu/dataset/280/higgs)
- **Shape:** 11M rows × 28 numeric features, binary label
- **Split:** first 10.5M rows train, last 500k rows test (canonical literature split)
- **Format:** Parquet, all features `float32`, label `int8`

## Model

- **Algorithm:** XGBoost gradient-boosted trees (`tree_method="hist"`)
- **Target hyperparameters:** 500 trees, max_depth 8, learning_rate 0.1, seed 42
- **Objective:** `binary:logistic`
- **Output:** positive-class probability ∈ [0, 1]

Concrete training metrics (AUC, accuracy, wall-clock train time, exact xgboost
version) are logged to `artifacts/metrics.json` after each `make train`.

## Export

- **Format:** ONNX, opset 15
- **Input tensor:** `features` — float32, shape `[None, 28]`
- **Output tensors:** `label` (int64, [N]), `probabilities` (float32, [N, 2])
- **Conversion:** `onnxmltools.convert_xgboost` with `zipmap=False`
- **Parity:** XGBoost ↔ ONNX max-abs probability diff verified `< 1e-5` over a
  10k-row test sample. Recorded in `artifacts/schema.json`.

## Reproducibility

- Python 3.11, dependencies pinned in `project/1-model/pyproject.toml` + `uv.lock`.
- Random seed 42 fixed for both training and the parity-check sampler.
- `model.onnx` SHA256 recorded in `schema.json` so both servers (Phase 2) load
  the byte-exact same artifact.

## Reproduce

```bash
make setup-model
make download   # ~7.5 GB; one-time
make train
make export
```

Or, for a quick pipeline smoke test (no real download, ~30 seconds):

```bash
make setup-model
make smoke
```
