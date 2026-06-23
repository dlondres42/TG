"""Acquire HIGGS dataset and produce train/test Parquet files.

HIGGS: 11M rows, 28 features, binary classification.
Source: UCI ML Repository (https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz)
Schema: 1 label column + 21 low-level + 7 high-level kinematic features.

Canonical literature split: first 10.5M rows train, last 500k rows test.

Modes:
  default              : download HIGGS.csv.gz, decompress, write train/test parquet
  --max-rows N         : truncate the read to N rows total (smoke testing)
  --synthetic N        : skip the network entirely; generate N synthetic rows with
                         the same shape as HIGGS. Used to validate the train/export
                         pipeline before committing to the 7.5 GB download.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
HIGGS_COLUMNS = ["label"] + [f"f{i}" for i in range(28)]
N_TRAIN_FULL = 10_500_000
N_TEST_FULL = 500_000

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        size_gb = dest.stat().st_size / 1e9
        print(f"[download] cached: {dest} ({size_gb:.2f} GB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="HIGGS"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))


def split_real(csv_gz: Path, max_rows: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_rows is None:
        n_total = N_TRAIN_FULL + N_TEST_FULL
        n_train = N_TRAIN_FULL
    else:
        n_total = max_rows
        # preserve the 21:1 train:test ratio of the canonical split
        n_train = max(1, int(n_total * N_TRAIN_FULL / (N_TRAIN_FULL + N_TEST_FULL)))

    print(f"[split] reading {n_total:,} rows from {csv_gz}")
    df = pd.read_csv(csv_gz, names=HIGGS_COLUMNS, nrows=n_total, dtype="float32")
    df["label"] = df["label"].astype("int8")
    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)
    print(f"[split] train: {len(train):,} rows  test: {len(test):,} rows")
    return train, test


def make_synthetic(n_total: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate HIGGS-shaped synthetic data for smoke testing.

    The label is a deterministic function of the first 3 features so that XGBoost
    can fit it cleanly — this exercises the full train/export pipeline without
    needing the real 7.5 GB download.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_total, 28)).astype("float32")
    y = (X[:, :3].sum(axis=1) + 0.1 * rng.standard_normal(n_total) > 0).astype("int8")

    n_train = max(1, int(n_total * N_TRAIN_FULL / (N_TRAIN_FULL + N_TEST_FULL)))
    df = pd.DataFrame(X, columns=HIGGS_COLUMNS[1:])
    df.insert(0, "label", y)
    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)
    print(f"[synthetic] train: {len(train):,} rows  test: {len(test):,} rows")
    return train, test


def write_parquet(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_out = DATA_DIR / "train.parquet"
    test_out = DATA_DIR / "test.parquet"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(train_out, index=False, engine="pyarrow")
    test.to_parquet(test_out, index=False, engine="pyarrow")
    print(f"[write] {train_out}  ({train_out.stat().st_size / 1e6:.1f} MB)")
    print(f"[write] {test_out}  ({test_out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--max-rows", type=int, default=None,
                   help="Limit total rows read from HIGGS.csv.gz (smoke testing).")
    g.add_argument("--synthetic", type=int, metavar="N", default=None,
                   help="Skip download; generate N synthetic rows with HIGGS shape.")
    args = p.parse_args()

    if args.synthetic is not None:
        train, test = make_synthetic(args.synthetic)
    else:
        csv_gz = DATA_DIR / "HIGGS.csv.gz"
        download_file(DATA_URL, csv_gz)
        train, test = split_real(csv_gz, args.max_rows)

    write_parquet(train, test)


if __name__ == "__main__":
    main()
