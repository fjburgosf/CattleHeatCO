"""Statistical helpers. Seed fixed at 20261004."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats

SEED = 20261004


def sen_slope(y: np.ndarray, x: np.ndarray | None = None) -> float:
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, float)
    i, j = np.triu_indices(len(y), 1)
    return float(np.median((y[j] - y[i]) / (x[j] - x[i])))


def trend_test(y: np.ndarray, alpha: float = 0.05) -> dict:
    """Mann-Kendall with the Hamed-Rao variance correction when the lag-1
    autocorrelation of the detrended series is significant; Sen slope per step."""
    y = np.asarray(y, float)
    n = len(y)
    slope = sen_slope(y)
    resid = y - slope * np.arange(n)
    r1 = pd.Series(resid).autocorr(lag=1)
    r1 = 0.0 if np.isnan(r1) else r1
    sig_ac = abs(r1) > stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    if np.ptp(y) == 0:
        return {"test": "constant", "p": 1.0, "tau": 0.0, "slope": 0.0, "lag1_r": r1}
    res = mk.hamed_rao_modification_test(y) if sig_ac else mk.original_test(y)
    return {"test": "hamed_rao" if sig_ac else "mk", "p": float(res.p), "tau": float(res.Tau),
            "slope": slope, "lag1_r": float(r1)}


def block_bootstrap_sen_ci(y: np.ndarray, block: int = 5, reps: int = 2000,
                           seed: int = SEED, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of the Sen slope from a moving-block bootstrap of residuals."""
    y = np.asarray(y, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    slope = sen_slope(y)
    fit = np.median(y - slope * np.arange(n)) + slope * np.arange(n)
    resid = y - fit
    starts = np.arange(n - block + 1)
    out = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, int(np.ceil(n / block)))])[:n]
        out[r] = sen_slope(fit + resid[idx])
    a = (1 - level) / 2
    return float(np.quantile(out, a)), float(np.quantile(out, 1 - a))


def bh_fdr(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: boolean array of rejected hypotheses."""
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    rej = np.zeros(len(p), bool)
    ps = p[ok]
    order = np.argsort(ps)
    m = len(ps)
    thresh = q * np.arange(1, m + 1) / m
    below = ps[order] <= thresh
    if below.any():
        k = np.max(np.nonzero(below)[0])
        r = np.zeros(m, bool)
        r[order[: k + 1]] = True
        rej[np.nonzero(ok)[0]] = r
    return rej


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def permutation_diff(x: np.ndarray, group: np.ndarray, reps: int = 10000, seed: int = SEED) -> tuple[float, float]:
    """Difference in means (group True minus False) and two-sided permutation p."""
    x, group = np.asarray(x, float), np.asarray(group, bool)
    obs = x[group].mean() - x[~group].mean()
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(reps):
        g = rng.permutation(group)
        cnt += abs(x[g].mean() - x[~g].mean()) >= abs(obs) - 1e-12
    return float(obs), float((cnt + 1) / (reps + 1))
