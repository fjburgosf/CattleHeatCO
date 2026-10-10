"""Detect department-years whose municipal values look permuted across rows.

For each department d and year y, municipal totals are compared (Spearman rank
correlation, log scale) with the mean of the adjacent census years. A genuine
census keeps municipal ranks very stable from year to year; a correlation far
below the department's usual level indicates misaligned rows. The department
total is compared with its neighbours too, because a permutation preserves it.
Output: results/quality_control/ica_alignment_by_dept_year.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
QC = ROOT / "results" / "quality_control"

p = pd.read_csv(ROOT / "data" / "processed" / "ica_bovine_panel.csv", dtype={"code": str})
p["dcode"] = p["code"].str[:2]
w = p.pivot_table(index=["dcode", "code"], columns="year", values="total", aggfunc="sum")
years = sorted(w.columns)
rows = []
for d, sub in w.groupby(level="dcode"):
    for y in years:
        nb = [k for k in (y - 1, y + 1) if k in years]
        ref = sub[nb].mean(axis=1)
        ok = sub[y].notna() & ref.notna()
        x, r = np.log1p(sub.loc[ok, y]), np.log1p(ref[ok])
        rho = spearmanr(x, r).statistic if ok.sum() >= 4 else np.nan
        mad = float(np.median(np.abs(x - r))) if ok.sum() else np.nan
        rows.append({"dcode": d, "year": y, "n_mun": int(ok.sum()), "spearman_vs_neighbours": rho,
                     "median_abs_log_diff": mad,
                     "dept_total": sub[y].sum(), "dept_total_neighbours_mean": sub[nb].sum().mean()})
out = pd.DataFrame(rows)
out["dept_total_ratio"] = out["dept_total"] / out["dept_total_neighbours_mean"]
# Flag: rank correlation below 0.80, or median absolute log change above log(1.5).
out["flag_misaligned"] = (out["spearman_vs_neighbours"] < 0.80) | (out["median_abs_log_diff"] > np.log(1.5))
out.to_csv(QC / "ica_alignment_by_dept_year.csv", index=False)
print(out[out.flag_misaligned].to_string(index=False))
print(out.groupby("year")["flag_misaligned"].sum())
