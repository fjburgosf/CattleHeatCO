"""Cell -> municipality weights on the ERA5-Land 0,1 degree grid.

For every municipality i (MGN 2024) and grid cell c the exact covered area a_ci
(km2) is computed with exactextract. Weights:
  area    : w_ci = a_ci / sum_c a_ci
  grazing : w_ci = a_ci p_c / sum_c a_ci p_c, p_c = grazing-compatible fraction
            (MapBiomas classes 21, 12, 11; mean 2017, 2020, 2024)
  mosaic  : as grazing with class 21 only (sensitivity)
Only ERA5-Land land cells (non-missing 2 m temperature) are used. A municipality
without any land cell takes its nearest land cell (listed in the QC output).
A municipality with grazing-compatible land below 1 % of its area falls back to
area weights.
Output: data/interim/weights/weights_long.csv, results/quality_control/weights_qc.csv
"""
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from exactextract import exact_extract
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
LAT_TOP, LON_LEFT, STEP, NLAT, NLON = 13.65, -81.85, 0.1, 180, 151
R_EARTH = 6371.0088


def grid_centres():
    lat = np.round(LAT_TOP - STEP / 2 - STEP * np.arange(NLAT), 2)
    lon = np.round(LON_LEFT + STEP / 2 + STEP * np.arange(NLON), 2)
    return lat, lon


def land_mask(era5_files: list[Path]) -> np.ndarray:
    """True where ERA5-Land has data, mapped onto the combined grid."""
    lat, lon = grid_centres()
    mask = np.zeros((NLAT, NLON), bool)
    for f in era5_files:
        ds = xr.open_dataset(f)
        tname = "valid_time" if "valid_time" in ds.dims else "time"
        t2m = ds["t2m"].isel({tname: 0})
        ok = t2m.notnull().values
        la = np.round(ds["latitude"].values, 2)
        lo = np.round(ds["longitude"].values, 2)
        iy = np.searchsorted(-lat, -la)
        ix = np.searchsorted(lon, lo)
        assert np.allclose(lat[iy], la) and np.allclose(lon[ix], lo), "grid mismatch"
        mask[np.ix_(iy, ix)] |= ok
    return mask


def main(era5_files: list[Path]) -> None:
    lat, lon = grid_centres()
    mask = land_mask(era5_files)
    frac = xr.open_dataset(ROOT / "data/interim/mapbiomas/grazing_fraction_0p1.nc")
    assert np.allclose(frac.latitude, lat) and np.allclose(frac.longitude, lon)
    graz = np.nan_to_num(frac["grazing_fraction_mean"].values)
    mosa = np.nan_to_num(frac["mosaic_fraction_mean"].values)

    ids = np.arange(NLAT * NLON, dtype=np.int32).reshape(NLAT, NLON)
    profile = dict(driver="GTiff", height=NLAT, width=NLON, count=1, dtype="int32",
                   crs="EPSG:4326", transform=from_origin(LON_LEFT, LAT_TOP, STEP, STEP))
    mun = gpd.read_file(f"zip://{ROOT / 'data/raw/dane/MGN2024_MPIO_POLITICO.zip'}!MGN_ADM_MPIO_GRAFICO.shp")
    mun = mun.to_crs(4326)[["mpio_cdpmp", "geometry"]].rename(columns={"mpio_cdpmp": "code"})
    with MemoryFile() as mf:
        with mf.open(**profile) as dst:
            dst.write(ids, 1)
        with mf.open() as src:
            res = exact_extract(src, mun, ["cell_id", "coverage", "values"],
                                include_cols=["code"], output="pandas")
    rows = []
    for _, r in res.iterrows():
        for v, cov in zip(r["values"], r["coverage"]):
            rows.append((r["code"], int(v), float(cov)))
    w = pd.DataFrame(rows, columns=["code", "cell", "coverage"])
    w["iy"], w["ix"] = np.divmod(w["cell"], NLON)
    w["lat"], w["lon"] = lat[w["iy"]], lon[w["ix"]]
    cell_km2 = (np.deg2rad(STEP) * R_EARTH) ** 2 * np.cos(np.deg2rad(w["lat"]))
    w["area_km2"] = w["coverage"] * cell_km2
    w["land"] = mask[w["iy"], w["ix"]]
    w["p_grazing"], w["p_mosaic"] = graz[w["iy"], w["ix"]], mosa[w["iy"], w["ix"]]

    qc, out = [], []
    land_iy, land_ix = np.nonzero(mask)
    for code, g in w.groupby("code"):
        if code.startswith("88"):  # San Andres y Providencia: no ERA5-Land land cell
            qc.append({"code": code, "n_land_cells": 0, "land_km2": 0.0, "grazing_share": np.nan,
                       "note": "excluded_no_era5land_land_cell"})
            continue
        g = g[g["land"]].copy()
        note = ""
        if g.empty:  # nearest land cell to the polygon's representative point
            pt = mun.loc[mun["code"] == code, "geometry"].iloc[0].representative_point()
            d = (lat[land_iy] - pt.y) ** 2 + (lon[land_ix] - pt.x) ** 2
            k = int(np.argmin(d))
            g = pd.DataFrame({"code": [code], "iy": [land_iy[k]], "ix": [land_ix[k]],
                              "lat": [lat[land_iy[k]]], "lon": [lon[land_ix[k]]], "area_km2": [1.0],
                              "p_grazing": [graz[land_iy[k], land_ix[k]]],
                              "p_mosaic": [mosa[land_iy[k], land_ix[k]]]})
            note = "nearest_land_cell"
        g["w_area"] = g["area_km2"] / g["area_km2"].sum()
        share_graz = (g["area_km2"] * g["p_grazing"]).sum() / g["area_km2"].sum()
        for key, p in (("w_grazing", "p_grazing"), ("w_mosaic", "p_mosaic")):
            num = g["area_km2"] * g[p]
            if num.sum() / g["area_km2"].sum() < 0.01:
                g[key] = g["w_area"]
                note += f";{key}_fallback_area"
            else:
                g[key] = num / num.sum()
        out.append(g[["code", "iy", "ix", "lat", "lon", "area_km2", "w_area", "w_grazing", "w_mosaic"]])
        qc.append({"code": code, "n_land_cells": len(g), "land_km2": g["area_km2"].sum(),
                   "grazing_share": share_graz, "note": note.strip(";")})
    out = pd.concat(out)
    (ROOT / "data/interim/weights").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "data/interim/weights/weights_long.csv", index=False)
    pd.DataFrame(qc).to_csv(ROOT / "results/quality_control/weights_qc.csv", index=False)
    q = pd.DataFrame(qc)
    print("municipalities:", q.shape[0], "| nearest-cell:", q.note.str.contains("nearest").sum(),
          "| grazing fallback:", q.note.str.contains("w_grazing_fallback").sum())


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
