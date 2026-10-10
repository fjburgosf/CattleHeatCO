"""Municipal monthly precipitation (CHIRPS v2.0) and SPI-3 / SPI-6.

Municipal precipitation is the area-weighted mean of CHIRPS 0,05 degree cells
(exact coverage fractions, MGN 2024). SPI follows the standard gamma procedure:
k-month sums, a two-parameter gamma fitted by maximum likelihood (location fixed
at 0) for each municipality and calendar month on the 1991-2020 baseline, a mixed
distribution H(x) = q + (1 - q) G(x) with q the baseline share of zero sums, and
SPI = inverse standard normal of H.
Output: data/processed/chirps_spi_mun_monthly.csv, results/quality_control/spi_qc.csv
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from exactextract import exact_extract
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/chirps/chirps_v2p0_monthly_colombia_198101_202512.nc"
BASE = (1991, 2020)


def municipal_precip() -> pd.DataFrame:
    ds = xr.open_dataset(SRC, decode_times=False)
    pr = ds["precipitation"]  # mm/month, dims T, Y, X (Y ascending)
    t = pd.period_range("1981-01", periods=ds.sizes["T"], freq="M")
    assert abs(float(ds["T"][0]) - 252.5) < 1e-6, "T origin must be Jan 1981 (months since 1960-01)"
    y, x = ds["Y"].values, ds["X"].values
    step = float(np.round(x[1] - x[0], 4))
    flip = y[0] < y[-1]
    arr = pr.values[:, ::-1, :] if flip else pr.values
    ny, nx = arr.shape[1:]
    ids = np.arange(ny * nx, dtype=np.int32).reshape(ny, nx)
    top = float(y.max()) + step / 2
    profile = dict(driver="GTiff", height=ny, width=nx, count=1, dtype="int32", crs="EPSG:4326",
                   transform=from_origin(float(x.min()) - step / 2, top, step, step))
    mun = gpd.read_file(f"zip://{ROOT / 'data/raw/dane/MGN2024_MPIO_POLITICO.zip'}!MGN_ADM_MPIO_GRAFICO.shp")
    mun = mun.to_crs(4326)[["mpio_cdpmp", "geometry"]].rename(columns={"mpio_cdpmp": "code"})
    with MemoryFile() as mf:
        with mf.open(**profile) as dst:
            dst.write(ids, 1)
        with mf.open() as src:
            res = exact_extract(src, mun, ["values", "coverage"], include_cols=["code"], output="pandas")
    flat = arr.reshape(arr.shape[0], -1)
    out, qc = {}, []
    for _, r in res.iterrows():
        cid = np.asarray(r["values"], int)
        cov = np.asarray(r["coverage"], float)
        lat = top - step * (cid // nx + 0.5)
        wgt = cov * np.cos(np.deg2rad(lat))
        vals = flat[:, cid]
        ok = ~np.isnan(vals[0])
        if not ok.any():
            qc.append({"code": r["code"], "note": "no_chirps_cell"})
            continue
        out[r["code"]] = (vals[:, ok] * wgt[ok]).sum(1) / wgt[ok].sum()
        qc.append({"code": r["code"], "note": "", "n_cells": int(ok.sum())})
    pd.DataFrame(qc).to_csv(ROOT / "results/quality_control/spi_qc.csv", index=False)
    return pd.DataFrame(out, index=t)


def spi(series: pd.Series, k: int) -> pd.Series:
    s = series.rolling(k, min_periods=k).sum()
    res = pd.Series(np.nan, index=s.index)
    for m in range(1, 13):
        idx = s.index.month == m
        base = s[idx & (s.index.year >= BASE[0]) & (s.index.year <= BASE[1])].dropna()
        q = float((base <= 0).mean())
        pos = base[base > 0]
        a, _, scale = stats.gamma.fit(pos, floc=0)
        x = s[idx]
        h = q + (1 - q) * stats.gamma.cdf(x.clip(lower=0), a, loc=0, scale=scale)
        h = np.where(x <= 0, q / 2 if q > 0 else 1e-6, h)  # zero sums: midpoint of the mass at zero
        res[idx] = stats.norm.ppf(np.clip(h, 1e-6, 1 - 1e-6))
    return res


def main() -> None:
    p = municipal_precip()
    rows = []
    for code in p.columns:
        s3, s6 = spi(p[code], 3), spi(p[code], 6)
        rows.append(pd.DataFrame({"code": code, "year": p.index.year, "month": p.index.month,
                                  "precip_mm": p[code].values, "spi3": s3.values, "spi6": s6.values}))
    out = pd.concat(rows)
    out.to_csv(ROOT / "data/processed/chirps_spi_mun_monthly.csv", index=False)
    b = out[(out.year >= BASE[0]) & (out.year <= BASE[1])]
    print("municipalities:", out.code.nunique(), "| SPI-3 baseline mean/sd:",
          round(b.spi3.mean(), 3), round(b.spi3.std(), 3), "| share SPI-3<=-1:",
          round((b.spi3 <= -1).mean(), 3))


if __name__ == "__main__":
    main()
