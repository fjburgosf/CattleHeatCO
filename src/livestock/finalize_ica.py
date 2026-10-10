"""Build the analysis-ready municipal cattle series on the MGN 2024 geometry.

Steps:
1. Map units absent from MGN 2024 to their legal successor
   (configs/divipola_successors.csv).
2. Department-years whose municipal rows are permuted in the ICA file
   (configs/ica_flagged_dept_years.csv) keep the reported department total but
   their municipal values are set to missing.
3. Primary series ('cattle'): for flagged department-years the department total is
   allocated to municipalities with the mean municipal share of the listed clean
   reference years. 'cattle_strict' keeps the flagged values missing (sensitivity).
Output: data/processed/ica_cattle_analysis.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs"

p = pd.read_csv(ROOT / "data" / "processed" / "ica_bovine_panel.csv", dtype={"code": str})
succ = pd.read_csv(CFG / "divipola_successors.csv", dtype=str)
p["code"] = p["code"].replace(dict(zip(succ["old_code"], succ["new_code"])))
cats = ["f_lt1", "m_lt1", "f_1_2", "m_1_2", "f_2_3", "m_2_3", "f_gt3", "m_gt3"]
g = p.groupby(["year", "code"], as_index=False)[["total", "farms_total"] + cats].sum(min_count=1)
g["dcode"] = g["code"].str[:2]
g["adults_gt2"] = g[["f_2_3", "m_2_3", "f_gt3", "m_gt3"]].sum(axis=1, min_count=4)
# Categories with negative values are unusable for category-based sensitivity.
neg = (g[cats] < 0).any(axis=1)
g.loc[neg, "adults_gt2"] = np.nan

flags = pd.read_csv(CFG / "ica_flagged_dept_years.csv", dtype={"dcode": str})
g["flag_misaligned"] = False
g["cattle"] = g["total"].astype(float)
g["cattle_strict"] = g["total"].astype(float)
g["allocation"] = "reported"
for _, f in flags.iterrows():
    y, d = int(f["year"]), f["dcode"]
    refs = [int(r) for r in str(f["reference_years_for_shares"]).split(";")]
    sel = (g["year"] == y) & (g["dcode"] == d)
    dept_total = g.loc[sel, "total"].sum()
    shares = []
    for r in refs:
        s = g[(g["year"] == r) & (g["dcode"] == d)].set_index("code")["total"]
        shares.append(s / s.sum())
    share = pd.concat(shares, axis=1).mean(axis=1)
    g.loc[sel, "flag_misaligned"] = True
    g.loc[sel, "cattle_strict"] = np.nan
    g.loc[sel, "cattle"] = g.loc[sel, "code"].map(share).values * dept_total
    g.loc[sel, "allocation"] = "dept_total_x_mean_share_" + "_".join(map(str, refs))
    g.loc[sel, "adults_gt2"] = np.nan

out = g[["year", "code", "dcode", "cattle", "cattle_strict", "total", "adults_gt2", "farms_total",
         "flag_misaligned", "allocation"]].rename(columns={"total": "cattle_reported"})
out.to_csv(ROOT / "data" / "processed" / "ica_cattle_analysis.csv", index=False)
chk = out.groupby("year")[["cattle", "cattle_reported"]].sum()
assert np.allclose(chk["cattle"], chk["cattle_reported"]), "allocation must preserve totals"
print(out.groupby("year").agg(units=("code", "nunique"), cattle=("cattle", "sum"),
                              misaligned_units=("flag_misaligned", "sum")))
