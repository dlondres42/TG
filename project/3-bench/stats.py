"""Phase 3 statistical add-ons computed from existing result.bin files.

All inputs are per-cell latency arrays (ns) already produced by `lib.vegeta_latency_status_ns`.
No bench rig is required — this module is pure post-processing over data on disk.

Provides:
  - bootstrap_percentile_ci: percentile CI on a single latency sample via resampling.
  - repeat_cov: across-repeat coefficient of variation for any per-cell scalar.
  - mann_whitney: two-sided MWU on two latency samples (SciPy normal approximation).
  - cliffs_delta: rank-based effect size in [-1, 1], with magnitude bucket.
  - little_law_in_flight: Little's Law n = lambda * W given RPS achieved + mean latency.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------- #
# Bootstrap CIs on percentiles
# --------------------------------------------------------------------------- #
def bootstrap_percentile_ci(
    sample: np.ndarray, q: float, n_boot: int = 1000, alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_lo, ci_hi) for the q-th percentile of `sample`.

    Resampling with replacement; n_boot=1000 gives a ~3% MC error on the CI
    endpoints which is well below run-to-run variance for our cells.
    """
    if rng is None:
        rng = np.random.default_rng(0xC0FFEE)
    sample = np.asarray(sample)
    if sample.size == 0:
        return (math.nan, math.nan, math.nan)
    point = float(np.percentile(sample, q))
    n = sample.size
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.percentile(sample[idx], q)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (point, lo, hi)


# --------------------------------------------------------------------------- #
# Across-repeat variability
# --------------------------------------------------------------------------- #
def repeat_cov(values: Iterable[float]) -> float:
    """Coefficient of variation (std/mean) across repeats. NaN if mean ~ 0."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size < 2:
        return math.nan
    m = float(arr.mean())
    if m == 0 or not math.isfinite(m):
        return math.nan
    return float(arr.std(ddof=1) / m)


# --------------------------------------------------------------------------- #
# Mann-Whitney U + Cliff's delta
# --------------------------------------------------------------------------- #
def mann_whitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided MWU. Returns (U_statistic, p_value).

    Uses SciPy's asymptotic approximation; exact at our sample sizes (>1000).
    """
    from scipy.stats import mannwhitneyu
    res = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    return float(res.statistic), float(res.pvalue)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Cliff's delta = P(a>b) - P(a<b) in [-1, 1], plus a magnitude bucket.

    Computed via SciPy's MWU statistic to avoid the O(n_a * n_b) pairwise count.
    Buckets follow Romano et al. 2006: negligible <0.147, small <0.33,
    medium <0.474, large otherwise.
    """
    from scipy.stats import mannwhitneyu
    res = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return (math.nan, "n/a")
    delta = (2.0 * res.statistic) / (n_a * n_b) - 1.0
    mag = abs(delta)
    if mag < 0.147:
        label = "negligible"
    elif mag < 0.33:
        label = "small"
    elif mag < 0.474:
        label = "medium"
    else:
        label = "large"
    return (float(delta), label)


# --------------------------------------------------------------------------- #
# Little's Law check
# --------------------------------------------------------------------------- #
def little_law_in_flight(rps_achieved: float, mean_latency_ms: float) -> float:
    """L = lambda * W. With lambda in req/s and W in seconds, L is mean
    concurrent in-flight request count. Compare against the server's effective
    parallelism (workers * threadpool or workers * intra_op_threads) to spot
    saturation: if L approaches or exceeds the budget, the queue is filling.
    """
    if not math.isfinite(rps_achieved) or not math.isfinite(mean_latency_ms):
        return math.nan
    return float(rps_achieved * (mean_latency_ms / 1000.0))


# --------------------------------------------------------------------------- #
# CCDF helper
# --------------------------------------------------------------------------- #
def ccdf_points(sample: np.ndarray, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, ccdf) suitable for a log-log tail plot.

    Down-samples to `max_points` evenly-spaced ranks so plots render fast even
    on 1M-sample inputs. The tail (top 1%) is kept dense — it's the part the
    P99/P99.9 story actually rides on.
    """
    sample = np.asarray(sample)
    if sample.size == 0:
        return (np.array([]), np.array([]))
    s = np.sort(sample)
    n = s.size
    if n <= max_points:
        ranks = np.arange(1, n + 1)
        return (s, 1.0 - (ranks - 1) / n)
    # Body: log-spaced from rank 1 to rank n*0.99.
    # Tail: dense linear from n*0.99 to n.
    body_end = int(n * 0.99)
    body_idx = np.unique(np.geomspace(1, body_end, num=max_points // 2).astype(int) - 1)
    tail_idx = np.arange(body_end, n)
    idx = np.unique(np.concatenate([body_idx, tail_idx]))
    return (s[idx], 1.0 - idx / n)
