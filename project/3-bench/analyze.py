"""Phase 3 analysis — aggregate L1/L2/L3 results across pipeline variants.

Reads results/layer1, results/layer2, results/layer3 and produces:
  results/summary.csv   one row per L3 run, with per-cell percentiles incl. P99.9
                        and goodput@{50, 100}ms (a fair cross-pipeline metric)
  results/budget.csv    per-variant latency budget at the (1,2) cell, BUDGET_RPS
  results/figures/      budget_stacked_bar, inflection, jitter, goodput_vs_rps

A "variant" is one row of the matrix-with-config-axis: `fastapi`,
`bentoml_lat5`, `bentoml_lat50`, `bentoml_lat250`, etc. Variants are discovered
from directory names under results/layer{2,3}/.

P99.9 and goodput@SLA are computed once per cell by encoding result.bin via a
Vegeta container, then cached into the cell's summary.json so repeat analyses
are fast.

Usage:
    uv run python analyze.py
    uv run python analyze.py --no-encode   # skip the (slow) per-cell bin encode
    uv run python analyze.py --sla-ms 50 100 200
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import stats as st  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = RESULTS / "figures"
BUDGET_RPS = 150
DEFAULT_SLA_MS = [50, 100]

# Cells where pairwise effect-size + bootstrap CI tables are emitted.
# (w, t, rps) tuples — picked to span low-load, mid-load, and saturation.
EFFECT_CELLS = [(1, 2, 150), (1, 2, 350), (1, 2, 600), (2, 2, 600), (2, 2, 1000)]
CCDF_CELL = (1, 2, 600)   # representative saturation cell for the CCDF figure


def augment_cell(cell_dir: Path, sla_ms: list[int]) -> None:
    """Cache P99.9 and goodput@SLA into the cell's summary.json (idempotent)."""
    spath = cell_dir / "summary.json"
    bin_path = cell_dir / "result.bin"
    if not spath.exists() or not bin_path.exists():
        return
    s = json.loads(spath.read_text())
    need_p999 = "p99_9_ms" not in s
    need_goodput = any(f"goodput_{x}ms" not in s for x in sla_ms)
    if not (need_p999 or need_goodput):
        return
    rows = lib.vegeta_latency_status_ns(cell_dir)
    if not rows:
        return
    lats_ns = [r[0] for r in rows]
    success_ok = [r for r in rows if 200 <= r[1] < 300]
    total = len(rows)
    if need_p999:
        s["p99_9_ms"] = float(np.percentile(lats_ns, 99.9)) / 1e6
    for x in sla_ms:
        key = f"goodput_{x}ms"
        if key not in s:
            limit_ns = x * 1_000_000
            within = sum(1 for r in success_ok if r[0] <= limit_ns)
            s[key] = within / total if total else 0.0
    spath.write_text(json.dumps(s, indent=2))


def load_l3(sla_ms: list[int], do_encode: bool, root_subdir: str = "layer3") -> pd.DataFrame:
    """Walk a layer3* subdirectory and assemble one row per cell run.

    `root_subdir` defaults to "layer3" (the main dataset); pass
    "layer3_with_cpu" to aggregate the mechanism subset that the overnight
    CPU-instrumented re-run produces.
    """
    rows = []
    layer3 = RESULTS / root_subdir
    if not layer3.exists():
        return pd.DataFrame()
    for variant_dir in sorted(p for p in layer3.iterdir() if p.is_dir()):
        variant_label = variant_dir.name
        for sjson in variant_dir.rglob("summary.json"):
            if do_encode:
                augment_cell(sjson.parent, sla_ms)
            s = json.loads(sjson.read_text())
            s.setdefault("variant", variant_label)
            s.pop("applied", None)
            s.pop("status_codes", None)
            rows.append(s)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["variant", "workers", "threads", "rps_target", "repeat"])
    # Little's Law: mean in-flight concurrency = rps_achieved * mean_latency.
    # Compares against effective worker*threadpool to spot saturation.
    if "rps_achieved" in df.columns and "mean_ms" in df.columns:
        df["little_law_n_in_flight"] = [
            st.little_law_in_flight(r, m) for r, m in zip(df["rps_achieved"], df["mean_ms"])
        ]
    return df


def build_budget(df_l3: pd.DataFrame) -> pd.DataFrame:
    l1_path = RESULTS / "layer1" / "L1_summary.json"
    if not l1_path.exists():
        return pd.DataFrame()
    l1 = json.loads(l1_path.read_text())
    l1_p50 = l1["results"]["1"]["per_call_ms"]["p50"]

    rows = []
    layer2 = RESULTS / "layer2"
    if not layer2.exists():
        return pd.DataFrame()
    for vdir in sorted(p for p in layer2.iterdir() if p.is_dir()):
        l2_path = vdir / "summary.json"
        if not l2_path.exists():
            continue
        variant = vdir.name
        l2 = json.loads(l2_path.read_text())
        l2_p50 = l2["p50_ms"]
        sub = df_l3[(df_l3["variant"] == variant)
                    & (df_l3["workers"] == 1) & (df_l3["threads"] == 2)
                    & (df_l3["rps_target"] == BUDGET_RPS)]
        l3_p50 = float(np.median(sub["p50_ms"])) if len(sub) else None
        rows.append({
            "variant": variant,
            "layer1_p50_ms": round(l1_p50, 4),
            "layer2_p50_ms": round(l2_p50, 4),
            "layer3_p50_ms": round(l3_p50, 4) if l3_p50 is not None else None,
            "framework_overhead_ms": round(l2_p50 - l1_p50, 4),
            "docker_overhead_ms": round(l3_p50 - l2_p50, 4) if l3_p50 is not None else None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_budget(budget: pd.DataFrame) -> None:
    """Grouped bars per variant: L1 (pure inference) vs L2 (host HTTP) vs L3 (containerised)."""
    if budget.empty:
        return
    plt = _mpl()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                         "legend.fontsize": 9})
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(budget))
    w = 0.27
    ax.bar(x - w, budget["layer1_p50_ms"],  w, label="L1 — pure inference",       color="#4C72B0")
    ax.bar(x,     budget["layer2_p50_ms"],  w, label="L2 — host HTTP (no Docker)",color="#DD8452")
    ax.bar(x + w, budget["layer3_p50_ms"],  w, label="L3 — containerised",        color="#55A868")
    for i, r in budget.reset_index(drop=True).iterrows():
        for off, key in [(-w, "layer1_p50_ms"), (0, "layer2_p50_ms"), (+w, "layer3_p50_ms")]:
            v = r[key]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                ax.text(i + off, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(budget["variant"], rotation=10, ha="right")
    ax.set_ylabel("P50 latency (ms)")
    ax.set_title(f"Latency budget at (1 worker, 2 threads), {BUDGET_RPS} RPS")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES / "budget_stacked_bar.png", dpi=120)
    plt.close(fig)


_VARIANT_STYLE = {
    "fastapi":         dict(color="#2E86AB", marker="o", lw=2.0, ms=7),
    "bentoml_lat5":    dict(color="#E63946", marker="s", lw=1.6, ms=6),
    "bentoml_lat50":   dict(color="#F4A261", marker="^", lw=1.6, ms=6),
    "bentoml_lat250":  dict(color="#7B2CBF", marker="D", lw=1.6, ms=6),
}


def _per_cell_plot(df: pd.DataFrame, metric_col: str, ylabel: str, title: str, fname: str,
                   ylog: bool = False, ylim: tuple | None = None) -> None:
    """2x2 grid (one panel per (workers, threads) cell), single shared legend.

    Sized for A4 print at \\textwidth: each panel stays >= 7 cm wide after
    LaTeX scaling, with fonts that survive the reduction.
    """
    plt = _mpl()
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
                         "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12})
    cells = sorted({(w, t) for w, t in zip(df["workers"], df["threads"])})
    ncols = 2
    nrows = (len(cells) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.6 * nrows),
                             squeeze=False, sharey=True, sharex=True)
    flat = axes.ravel()
    for ax in flat[len(cells):]:
        ax.set_visible(False)
    for ax, (w, t) in zip(flat, cells):
        sub = df[(df["workers"] == w) & (df["threads"] == t)]
        for variant in sorted(sub["variant"].unique()):
            g = sub[sub["variant"] == variant].groupby("rps_target")[metric_col].median()
            style = _VARIANT_STYLE.get(variant, dict(marker="o"))
            style = {**style, "lw": max(style.get("lw", 2.0), 2.2),
                     "ms": max(style.get("ms", 7), 8)}
            ax.plot(g.index, g.values, label=variant, **style)
        ax.set_title(f"workers={w}, threads={t}")
        ax.grid(True, which="both", alpha=0.3)
        if ylog:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(ylim)
    # Axis labels only on the outer edge (sharex/sharey keep panels aligned).
    for ax in axes[-1, :]:
        ax.set_xlabel("RPS alvo")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    # One legend for the whole figure instead of four duplicates.
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    if title:
        fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98 if title else 1.0))
    fig.savefig(FIGURES / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_inflection(df: pd.DataFrame) -> None:
    if df.empty:
        return
    # P99 spans 1ms (FastAPI) to >1000ms (BentoML saturation) — log scale needed.
    # Title left empty: the thesis/report caption carries the message.
    _per_cell_plot(df, "p99_ms", "latência P99 (ms, log)",
                   "", "inflection.png", ylog=True)


def fig_jitter(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df = df.copy()
    df["jitter"] = df["p99_ms"] / df["p50_ms"]
    # Jitter spans 2x (FastAPI) to 100x (BentoML at high load) — log scale clearer.
    _per_cell_plot(df, "jitter", "P99 / P50 (log)",
                   "", "jitter.png", ylog=True)


def fig_goodput(df: pd.DataFrame, sla_col: str, sla_ms: int) -> None:
    if df.empty or sla_col not in df.columns:
        return
    # Goodput is a fraction in [0,1] — fixed linear scale lets cliffs be read at a glance.
    _per_cell_plot(df, sla_col, f"vazão útil @ {sla_ms} ms (fração)",
                   "", f"goodput_{sla_ms}ms_vs_rps.png", ylim=(-0.02, 1.05))


def fig_replicate_box(df: pd.DataFrame) -> None:
    """Boxplot of the three replicate P99 values per cell.

    Raw points are overlaid on each box so no reader mistakes n=3 for a
    larger sample; the box conveys median + spread, the dots the truth.
    """
    if df.empty:
        return
    plt = _mpl()
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
                         "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12})
    from matplotlib.patches import Patch
    variants = [v for v in _VARIANT_STYLE if v in set(df["variant"])]
    rps_levels = sorted(df["rps_target"].unique())
    cells = sorted({(w, t) for w, t in zip(df["workers"], df["threads"])})
    ncols = 2
    nrows = (len(cells) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.6 * nrows),
                             squeeze=False, sharey=True, sharex=True)
    flat = axes.ravel()
    for ax in flat[len(cells):]:
        ax.set_visible(False)
    width = 0.19
    offsets = {v: (i - (len(variants) - 1) / 2) * width for i, v in enumerate(variants)}
    for ax, (w, t) in zip(flat, cells):
        sub = df[(df["workers"] == w) & (df["threads"] == t)]
        for variant in variants:
            color = _VARIANT_STYLE[variant]["color"]
            for xi, rps in enumerate(rps_levels):
                vals = sub[(sub["variant"] == variant)
                           & (sub["rps_target"] == rps)]["p99_ms"].dropna().values
                if len(vals) == 0:
                    continue
                pos = xi + offsets[variant]
                ax.boxplot([vals], positions=[pos], widths=width * 0.85,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color, lw=1.2),
                           whiskerprops=dict(color=color, lw=1.2),
                           capprops=dict(color=color, lw=1.2),
                           medianprops=dict(color=color, lw=1.8))
                ax.plot([pos] * len(vals), vals, linestyle="none", marker="o",
                        ms=3.5, color=color, zorder=3)
        ax.set_title(f"workers={w}, threads={t}")
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)
        ax.set_xticks(range(len(rps_levels)))
        ax.set_xticklabels([str(int(r)) for r in rps_levels])
    for ax in axes[-1, :]:
        ax.set_xlabel("RPS alvo")
    for ax in axes[:, 0]:
        ax.set_ylabel("latência P99 (ms, log)")
    handles = [Patch(facecolor=_VARIANT_STYLE[v]["color"], alpha=0.5,
                     edgecolor=_VARIANT_STYLE[v]["color"], label=v) for v in variants]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(variants), 4),
               framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1.0))
    fig.savefig(FIGURES / "replicate_box_p99.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-cell statistical tables (CIs, effect sizes) — read result.bin directly
# --------------------------------------------------------------------------- #
def _load_cell_latencies(cell_dir: Path) -> np.ndarray:
    rows = lib.vegeta_latency_status_ns(cell_dir)
    if not rows:
        return np.array([], dtype=np.float64)
    return np.array([r[0] for r in rows if 200 <= r[1] < 300], dtype=np.float64) / 1e6


@functools.lru_cache(maxsize=256)
def _gather_variant_latencies(variant_dir: Path, w: int, t: int, rps: int) -> np.ndarray:
    """Concatenate per-request latencies (ms, 2xx only) across all repeats of one cell.

    Cached because the CI table, effect-size table, and CCDF figure all read
    overlapping cells; without the cache we'd re-encode the same result.bin
    files 3x via the slow `vegeta encode` path.
    """
    cell_root = variant_dir / f"w{w}_t{t}" / f"rps{rps}"
    if not cell_root.exists():
        return np.array([], dtype=np.float64)
    chunks = [_load_cell_latencies(run) for run in sorted(cell_root.glob("run*"))]
    chunks = [c for c in chunks if c.size > 0]
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)


def build_ci_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, cell): bootstrap 95% CI on P50/P95/P99 + across-repeat CoV.

    The CoV column captures *run-to-run* variability (3 numbers); the bootstrap
    CI captures *within-pooled-sample* uncertainty on the headline percentile.
    Both are needed: a tight CI with high CoV means the runs are individually
    confident but disagree with each other (instability), not just sampling noise.
    """
    rows = []
    layer3 = RESULTS / "layer3"
    rng = np.random.default_rng(0xC0FFEE)
    for variant_dir in sorted(p for p in layer3.iterdir() if p.is_dir()):
        variant = variant_dir.name
        sub_v = df[df["variant"] == variant]
        for w, t, rps in EFFECT_CELLS:
            sub = sub_v[(sub_v["workers"] == w) & (sub_v["threads"] == t) & (sub_v["rps_target"] == rps)]
            if sub.empty:
                continue
            sample = _gather_variant_latencies(variant_dir, w, t, rps)
            if sample.size == 0:
                continue
            p50, p50_lo, p50_hi = st.bootstrap_percentile_ci(sample, 50, rng=rng)
            p95, p95_lo, p95_hi = st.bootstrap_percentile_ci(sample, 95, rng=rng)
            p99, p99_lo, p99_hi = st.bootstrap_percentile_ci(sample, 99, rng=rng)
            rows.append({
                "variant": variant, "workers": w, "threads": t, "rps_target": rps,
                "n_pooled": int(sample.size),
                "p50_ms": round(p50, 4), "p50_ci_lo": round(p50_lo, 4), "p50_ci_hi": round(p50_hi, 4),
                "p95_ms": round(p95, 4), "p95_ci_lo": round(p95_lo, 4), "p95_ci_hi": round(p95_hi, 4),
                "p99_ms": round(p99, 4), "p99_ci_lo": round(p99_lo, 4), "p99_ci_hi": round(p99_hi, 4),
                "p50_cov_across_repeats": round(st.repeat_cov(sub["p50_ms"]), 4),
                "p99_cov_across_repeats": round(st.repeat_cov(sub["p99_ms"]), 4),
            })
    return pd.DataFrame(rows)


def build_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Mann-Whitney U + Cliff's delta per cell.

    For each cell in EFFECT_CELLS, compare every variant pair on the pooled
    2xx latency sample. Cliff's delta names the direction (positive => left
    variant has larger latencies) and magnitude is bucketed per Romano 2006.
    """
    rows = []
    layer3 = RESULTS / "layer3"
    variants = sorted(p.name for p in layer3.iterdir() if p.is_dir())
    for w, t, rps in EFFECT_CELLS:
        samples = {}
        for v in variants:
            arr = _gather_variant_latencies(layer3 / v, w, t, rps)
            if arr.size > 0:
                samples[v] = arr
        names = sorted(samples.keys())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                u, p = st.mann_whitney(samples[a], samples[b])
                delta, mag = st.cliffs_delta(samples[a], samples[b])
                rows.append({
                    "workers": w, "threads": t, "rps_target": rps,
                    "variant_a": a, "variant_b": b,
                    "n_a": int(samples[a].size), "n_b": int(samples[b].size),
                    "mwu_U": round(u, 1), "mwu_p": float(f"{p:.4g}"),
                    "cliffs_delta": round(delta, 4), "magnitude": mag,
                })
    return pd.DataFrame(rows)


def fig_ccdf(df: pd.DataFrame) -> None:
    """CCDF (1 - CDF) tail per variant at CCDF_CELL, log-log scale.

    Standard HDR-histogram-style tail visualization: any heavy right tail
    looks like a straight or upward-curving line on log-log; a clean
    distribution falls off quickly. Highlights the difference between
    FastAPI (clipped tail) and BentoML (long, structured tail) at saturation.
    """
    layer3 = RESULTS / "layer3"
    if not layer3.exists():
        return
    w, t, rps = CCDF_CELL
    plt = _mpl()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                         "legend.fontsize": 9})
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    any_drawn = False
    for variant_dir in sorted(p for p in layer3.iterdir() if p.is_dir()):
        sample = _gather_variant_latencies(variant_dir, w, t, rps)
        if sample.size < 100:
            continue
        x, y = st.ccdf_points(sample)
        style = _VARIANT_STYLE.get(variant_dir.name, dict(color="#444", marker=None))
        ax.plot(x, y, label=variant_dir.name, color=style.get("color", "#444"), lw=1.6)
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("latency (ms, log)")
    ax.set_ylabel("P(latency > x)  (log)")
    ax.set_title(f"Tail CCDF at workers={w}, threads={t}, {rps} RPS (2xx only)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES / "ccdf_tail.png", dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-encode", action="store_true",
                   help="skip per-cell bin encode (P99.9 and goodput@SLA stay cached or missing)")
    p.add_argument("--sla-ms", nargs="+", type=int, default=DEFAULT_SLA_MS,
                   help="SLA thresholds (ms) for goodput")
    args = p.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)

    df = load_l3(sla_ms=args.sla_ms, do_encode=not args.no_encode)
    if df.empty:
        print("[analyze] no L3 results yet")
    else:
        df.to_csv(RESULTS / "summary.csv", index=False)
        print(f"[analyze] wrote summary.csv ({len(df)} runs, variants={sorted(df['variant'].unique())})")
        fig_inflection(df)
        fig_jitter(df)
        fig_replicate_box(df)
        for sla in args.sla_ms:
            col = f"goodput_{sla}ms"
            if col in df.columns:
                fig_goodput(df, col, sla)
        print(f"[analyze] wrote inflection.png, jitter.png, goodput_<sla>ms_vs_rps.png")

    budget = build_budget(df) if not df.empty else pd.DataFrame()
    if budget.empty:
        print(f"[analyze] budget skipped — need L1 + L2 + L3 (1,2)@{BUDGET_RPS} RPS for ≥1 variant")
    else:
        budget.to_csv(RESULTS / "budget.csv", index=False)
        fig_budget(budget)
        print("[analyze] wrote budget.csv, budget_stacked_bar.png")
        print(budget.to_string(index=False))

    if not df.empty:
        ci = build_ci_table(df)
        if not ci.empty:
            ci.to_csv(RESULTS / "ci.csv", index=False)
            print(f"[analyze] wrote ci.csv ({len(ci)} rows: bootstrap CIs + repeat CoV)")
        eff = build_effect_sizes(df)
        if not eff.empty:
            eff.to_csv(RESULTS / "effect_sizes.csv", index=False)
            print(f"[analyze] wrote effect_sizes.csv ({len(eff)} pairwise rows)")
        fig_ccdf(df)
        if (FIGURES / "ccdf_tail.png").exists():
            print(f"[analyze] wrote ccdf_tail.png (cell={CCDF_CELL})")

    # CPU-instrumented mechanism subset (overnight re-run). Stored separately
    # so the headline dataset stays untouched; rolled up into cpu_mechanism.csv
    # only when the layer3_with_cpu tree exists.
    if (RESULTS / "layer3_with_cpu").exists():
        df_cpu = load_l3(sla_ms=args.sla_ms, do_encode=False, root_subdir="layer3_with_cpu")
        if not df_cpu.empty:
            keep = ["variant", "workers", "threads", "rps_target", "repeat",
                    "rps_achieved", "error_rate",
                    "p50_ms", "p95_ms", "p99_ms", "mean_ms",
                    "cpu_mean_pct", "cpu_p50_pct", "cpu_p95_pct", "cpu_max_pct", "cpu_n_samples"]
            keep = [c for c in keep if c in df_cpu.columns]
            df_cpu[keep].to_csv(RESULTS / "cpu_mechanism.csv", index=False)
            print(f"[analyze] wrote cpu_mechanism.csv ({len(df_cpu)} CPU-instrumented runs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
