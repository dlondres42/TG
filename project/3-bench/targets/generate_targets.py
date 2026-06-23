"""Generate Vegeta request bodies from real HIGGS test rows.

Samples N rows from the Phase 1 test parquet and writes them as a JSON-lines
file (`bodies.jsonl`), one `{"features": [[...28...]]}` object per line. The
L2/L3 harness turns this into a per-pipeline Vegeta targets file (correct URL +
base64 body) via lib.build_targets.

Tree-ensemble inference latency is essentially data-independent, so the row
values don't change the measured latency — cycling through real rows simply
rules out any "you measured one cached path" objection and keeps payloads
realistic.

Usage:
    uv run python targets/generate_targets.py            # 1000 rows (default)
    uv run python targets/generate_targets.py --n 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # project/
TEST_PARQUET = ROOT / "1-model" / "data" / "test.parquet"
OUT = HERE / "bodies.jsonl"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1000, help="number of rows to sample")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not TEST_PARQUET.exists():
        print(f"[targets] missing {TEST_PARQUET}; run Phase 1 (make train) first", file=sys.stderr)
        return 1

    df = pd.read_parquet(TEST_PARQUET)
    feature_cols = [c for c in df.columns if c != "label"]
    n = min(args.n, len(df))
    sample = df.sample(n=n, random_state=args.seed)[feature_cols].astype("float32")

    with open(OUT, "w", encoding="utf-8") as f:
        for row in sample.to_numpy().tolist():
            f.write(json.dumps({"features": [row]}) + "\n")

    print(f"[targets] wrote {n} bodies -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
