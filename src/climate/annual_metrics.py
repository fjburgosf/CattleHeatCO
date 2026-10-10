"""Annual and monthly municipal heat metrics from the processed ERA5-Land outputs.

Inputs : data/interim/climate/mun_monthly_Y.csv, mun_hourly_thi_Y.nc (local time)
Outputs: data/processed/heat_mun_annual.csv   (code, year, weighting, DH_*, N_*,
                                               thi_mean, thi_dmax_max, thi_p99,
                                               days_dmax_ge79, maxrun_ge75_h)
         data/processed/heat_mun_monthly.csv  (code, year, month, DH_75 by weighting,
                                               mean daily max THI, any hour >= 75)
Durations are computed within each local calendar year (runs are not carried
across 31 December).
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "interim" / "climate"
OUT = ROOT / "data" / "processed"


def max_run(b: np.ndarray) -> np.ndarray:
    """Longest run of True along axis 0 for each column."""
    best = np.zeros(b.shape[1], int)
    cur = np.zeros(b.shape[1], int)
    for row in b:
        cur = np.where(row, cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def main(y0: int, y1: int) -> None:
    annual, monthly = [], []
    for y in range(y0, y1 + 1):
        m = pd.read_csv(IN / f"mun_monthly_{y}.csv", dtype={"code": str})
        a = m.groupby(["code", "weighting"]).sum(numeric_only=True).drop(columns="month").reset_index()
        a.insert(1, "year", y)
        h = xr.open_dataset(IN / f"mun_hourly_thi_{y}.nc")
        thi = h["thi"].values  # (time, code)
        t = pd.DatetimeIndex(h["time"].values)
        assert (t.year == y).all() and len(t) in (8760, 8784), f"incomplete local year {y}: {len(t)} h"
        codes = h["code"].values.astype(str)
        day = t.normalize()
        dmax = pd.DataFrame(thi, index=t).groupby(day).max()
        hourly_stats = pd.DataFrame({
            "code": codes, "thi_mean": thi.mean(0), "thi_dmax_max": dmax.max().values,
            "thi_p99": np.percentile(thi, 99, axis=0), "days_dmax_ge79": (dmax >= 79).sum().values,
            "maxrun_ge75_h": max_run(thi >= 75)})
        a = a.merge(hourly_stats, on="code", how="left")
        annual.append(a)
        dm = dmax.groupby(dmax.index.month).mean()
        any75 = pd.DataFrame(thi >= 75, index=t).groupby(t.month).any()
        piv = m.pivot_table(index=["code", "month"], columns="weighting", values="DH_75").add_prefix("DH75_")
        mm = pd.DataFrame({"code": np.repeat(codes[None, :], 12, 0).ravel(),
                           "month": np.repeat(np.arange(1, 13), len(codes)),
                           "thi_dmax_mean": dm.values.ravel(), "any_ge75": any75.values.ravel()})
        mm = mm.merge(piv.reset_index(), on=["code", "month"], how="left")
        mm.insert(1, "year", y)
        monthly.append(mm)
        print(y, flush=True)
    pd.concat(annual).to_csv(OUT / "heat_mun_annual.csv", index=False)
    pd.concat(monthly).to_csv(OUT / "heat_mun_monthly.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
