"""Hourly THI from ERA5-Land and its aggregation to municipalities.

For each local (UTC-5) month of the requested years:
  RH  = 100 e_s(Td)/e_s(T), e_s(x) = 6,1094 exp(17,625 x / (x + 243,04)), clipped to 100
  THI = (1,8 T + 32) - (0,55 - 0,0055 RH)(1,8 T - 26)            [T, Td in degC]
Cell-level monthly degree-hours DH_tau = sum max(0, THI - tau) and exceedance hours
N_tau are computed first and then averaged with the municipal weights (area,
grazing, mosaic). The hourly municipal THI series (grazing weights) is stored for
duration, extreme and compound metrics.

Outputs per year Y (data/interim/climate/):
  mun_monthly_Y.csv   code, month, weighting, DH_*, N_*
  mun_hourly_thi_Y.nc hourly THI (grazing weights), local time, float32
  cell_monthly_Y.nc   DH75 and N75 on land cells (maps)
  qc_Y.csv            completeness and range checks
Usage: python src/climate/process_era5land.py 1981 2025
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "era5land"
OUT = ROOT / "data" / "interim" / "climate"
THRESHOLDS = (68.8, 72.0, 75.0, 79.0)
LAT_TOP, LON_LEFT, STEP, NLAT, NLON = 13.65, -81.85, 0.1, 180, 151
UTC_OFFSET = pd.Timedelta(hours=5)  # local time = UTC - 5 h


def thi_from(t_k: np.ndarray, td_k: np.ndarray) -> tuple[np.ndarray, int]:
    t, td = t_k - 273.15, td_k - 273.15
    es = lambda x: 6.1094 * np.exp(17.625 * x / (x + 243.04))
    rh = 100.0 * es(td) / es(t)
    n_clip = int(np.count_nonzero(rh > 100.0))
    rh = np.minimum(rh, 100.0)
    return (1.8 * t + 32.0) - (0.55 - 0.0055 * rh) * (1.8 * t - 26.0), n_clip


def load_weights():
    w = pd.read_csv(ROOT / "data/interim/weights/weights_long.csv", dtype={"code": str})
    cells = np.unique(w["iy"].to_numpy() * NLON + w["ix"].to_numpy())
    pos = {c: k for k, c in enumerate(cells)}
    codes = np.array(sorted(w["code"].unique()))
    cpos = {c: k for k, c in enumerate(codes)}
    rows = np.array([pos[c] for c in w["iy"] * NLON + w["ix"]])
    cols = np.array([cpos[c] for c in w["code"]])
    mats = {k: sparse.csr_matrix((w[f"w_{k}"].to_numpy(), (rows, cols)), shape=(len(cells), len(codes)))
            for k in ("area", "grazing", "mosaic")}
    return cells, codes, mats


def read_hours(start: pd.Timestamp, end: pd.Timestamp, cells: np.ndarray):
    """Hourly T and Td (K) on the needed land cells for UTC [start, end)."""
    months = pd.period_range(start, end - pd.Timedelta(hours=1), freq="M")
    t_parts, d_parts, times = [], [], None
    for dom in ("main",):  # the island domain has no land cells
        tt, dd, tm = [], [], []
        for p in months:
            f = RAW / dom / f"era5land_{dom}_{p.year}{p.month:02d}.nc"
            ds = xr.open_dataset(f)
            tn = "valid_time" if "valid_time" in ds.dims else "time"
            ds = ds.sel({tn: slice(start, end - pd.Timedelta(hours=1))})
            la = np.round(ds["latitude"].values, 2)
            lo = np.round(ds["longitude"].values, 2)
            iy = np.rint((LAT_TOP - STEP / 2 - la) / STEP).astype(int)
            ix = np.rint((lo - (LON_LEFT + STEP / 2)) / STEP).astype(int)
            cid = (iy[:, None] * NLON + ix[None, :]).ravel()
            sel = np.isin(cid, cells)
            tt.append(ds["t2m"].values.reshape(ds.sizes[tn], -1)[:, sel])
            dd.append(ds["d2m"].values.reshape(ds.sizes[tn], -1)[:, sel])
            tm.append(ds[tn].values)
            ids = cid[sel]
        t_parts.append((ids, np.concatenate(tt)))
        d_parts.append((ids, np.concatenate(dd)))
        times = np.concatenate(tm) if times is None else times
    n = len(times)
    T = np.full((n, len(cells)), np.nan, np.float32)
    D = np.full((n, len(cells)), np.nan, np.float32)
    pos = np.searchsorted(cells, np.concatenate([p[0] for p in t_parts]))
    T[:, pos] = np.concatenate([p[1] for p in t_parts], axis=1)
    D[:, pos] = np.concatenate([p[1] for p in d_parts], axis=1)
    return times, T, D


def process_year(year: int, cells, codes, mats) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    monthly, hourly, hourly_t, cell_dh, cell_n, qc = [], [], [], [], [], []
    for m in range(1, 13):
        start = pd.Timestamp(year, m, 1) + UTC_OFFSET
        end = (pd.Timestamp(year, m, 1) + pd.offsets.MonthBegin(1)) + UTC_OFFSET
        times, T, D = read_hours(start, end, cells)
        expected = int((end - start) / pd.Timedelta(hours=1))
        thi, n_clip = thi_from(T, D)
        nan_frac = float(np.isnan(thi).mean())
        qc.append({"year": year, "month": m, "hours": len(times), "expected_hours": expected,
                   "nan_fraction": nan_frac, "rh_clipped": n_clip,
                   "td_gt_t": int(np.count_nonzero(D > T + 0.01)),
                   "thi_min": float(np.nanmin(thi)), "thi_max": float(np.nanmax(thi))})
        thi = np.nan_to_num(thi, nan=np.nanmean(thi))  # isolated gaps only; counted in QC
        rec = {}
        for tau in THRESHOLDS:
            ex = thi - tau
            dh = np.clip(ex, 0, None).sum(axis=0)
            nh = (ex >= 0).sum(axis=0).astype(np.float64)
            rec[f"DH_{tau:g}"], rec[f"N_{tau:g}"] = dh, nh
            if tau == 75.0:
                cell_dh.append(dh.astype(np.float32))
                cell_n.append(nh.astype(np.float32))
        for k, W in mats.items():
            df = pd.DataFrame({key: W.T @ v for key, v in rec.items()})
            df.insert(0, "weighting", k)
            df.insert(0, "month", m)
            df.insert(0, "code", codes)
            monthly.append(df)
        hourly.append((mats["grazing"].T @ thi.T).T.astype(np.float32))
        hourly_t.append(pd.DatetimeIndex(times) - UTC_OFFSET)
        print(year, m, len(times), f"nan={nan_frac:.4f}", flush=True)
    xr.Dataset({"thi": (("time", "code"), np.concatenate(hourly))},
               coords={"time": np.concatenate(hourly_t), "code": codes},
               attrs={"weighting": "grazing", "time": "local (UTC-5)"}).to_netcdf(
        OUT / f"mun_hourly_thi_{year}.nc", encoding={"thi": {"zlib": True, "complevel": 4}})
    xr.Dataset({"DH75": (("month", "cell"), np.stack(cell_dh)), "N75": (("month", "cell"), np.stack(cell_n))},
               coords={"month": np.arange(1, 13), "cell": cells}).to_netcdf(OUT / f"cell_monthly_{year}.nc")
    pd.DataFrame(qc).to_csv(OUT / f"qc_{year}.csv", index=False)
    # written last: its presence marks the year as complete (safe resume after shutdown)
    tmp = OUT / f"mun_monthly_{year}.part"
    pd.concat(monthly).to_csv(tmp, index=False)
    tmp.replace(OUT / f"mun_monthly_{year}.csv")


def main() -> None:
    y0, y1 = int(sys.argv[1]), int(sys.argv[2])
    cells, codes, mats = load_weights()
    for y in range(y0, y1 + 1):
        if (OUT / f"mun_monthly_{y}.csv").exists():
            continue
        process_year(y, cells, codes, mats)


if __name__ == "__main__":
    main()
