"""Train an XGBoost binary classifier on HIGGS train.parquet.

Outputs:
  artifacts/model.ubj      — XGBoost native binary format (preserves dtypes)
  artifacts/metrics.json   — AUC, accuracy, training time, hyperparameters
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet")
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-trees", type=int, default=500,
                   help="Number of boosting rounds (default: 500 — proposal target).")
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    print(f"[train] X_train={X_train.shape}  X_test={X_test.shape}")

    model = xgb.XGBClassifier(
        n_estimators=args.n_trees,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.seed,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_secs = time.time() - t0

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba > 0.5).astype(np.int32)

    metrics = {
        "auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "n_trees": args.n_trees,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "train_seconds": round(train_secs, 2),
        "xgboost_version": xgb.__version__,
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS / "model.ubj"
    model.save_model(str(model_path))
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[train] AUC={metrics['auc']:.4f}  acc={metrics['accuracy']:.4f}  "
          f"trained in {train_secs:.1f}s")
    print(f"[train] saved -> {model_path}")


if __name__ == "__main__":
    main()
