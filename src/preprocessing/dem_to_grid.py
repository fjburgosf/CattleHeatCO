"""Copernicus DEM GLO-90 -> 0,1 degree climate grid -> municipal mean elevation.

GLO-90 tiles (3 arc-second, 1200 x 1200 px per degree at these latitudes) are
averaged into 0,05 degree sub-cells (60 x 60 px) whose edges align with both the
tile edges and the ERA5-Land cell edges (cells centred on multiples of 0,1 degree).
Pairs of sub-cells are then averaged into 0,1 degree cells and combined with the
municipal weights (data/interim/weights/weights_long.csv).
Output: data/processed/dem_mun.csv (elev_area_mean_m, elev_grazing_mean_m), data/interim/dem_grid_0p1.nc
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]
TILES = ROOT / "data/raw/dem_glo90"
LAT0, LON0, NY, NX = 14.0, -80.0, 380, 280  # 0,05 degree sub-grid, top-left corner
LAT_TOP, LON_LEFT, STEP, NLAT, NLON = 13.65, -81.85, 0.1, 180, 151


def main() -> None:
    sub = np.full((NY, NX), np.nan)
    for f in sorted(TILES.glob("*.tif")):
        with rasterio.open(f) as src:
            a = src.read(1).astype(float)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            top, left = round(src.transform.f), round(src.transform.c)
            h, w = a.shape
            assert (h, w) == (1200, 1200), f"unexpected tile size {f.name}: {a.shape}"
            m = np.nanmean(a.reshape(20, 60, 20, 60), axis=(1, 3))
            r0 = int(round((LAT0 - top) / 0.05))
            c0 = int(round((left - LON0) / 0.05))
            sub[r0:r0 + 20, c0:c0 + 20] = m
    # 0,1 degree cell centred at (lat, lon) covers sub-cells with edges lat-0,05 ... lat+0,05
    grid = np.full((NLAT, NLON), np.nan)
    lat = np.round(LAT_TOP - STEP / 2 - STEP * np.arange(NLAT), 2)
    lon = np.round(LON_LEFT + STEP / 2 + STEP * np.arange(NLON), 2)
    for i, la in enumerate(lat):
        r = int(round((LAT0 - (la + 0.05)) / 0.05))
        if r < 0 or r + 2 > NY:
            continue
        for j, lo in enumerate(lon):
            c = int(round(((lo - 0.05) - LON0) / 0.05))
            if c < 0 or c + 2 > NX:
                continue
            blk = sub[r:r + 2, c:c + 2]
            if np.isfinite(blk).any():
                grid[i, j] = np.nanmean(blk)
    import xarray as xr
    xr.Dataset({"elev_m": (("latitude", "longitude"), grid)}, coords={"latitude": lat, "longitude": lon},
               attrs={"source": "Copernicus DEM GLO-90, mean per 0,1 degree cell"}).to_netcdf(
        ROOT / "data/interim/dem_grid_0p1.nc")
    w = pd.read_csv(ROOT / "data/interim/weights/weights_long.csv", dtype={"code": str})
    w["elev"] = grid[w["iy"], w["ix"]]
    ok = w["elev"].notna()
    out = w[ok].groupby("code").apply(lambda d: pd.Series({
        "elev_area_mean_m": np.average(d["elev"], weights=d["w_area"]),
        "elev_grazing_mean_m": np.average(d["elev"], weights=d["w_grazing"]) if d["w_grazing"].sum() > 0 else np.nan,
        "dem_cells": len(d)}))
    out.to_csv(ROOT / "data/processed/dem_mun.csv")
    print(out.describe().round(1))


if __name__ == "__main__":
    main()
