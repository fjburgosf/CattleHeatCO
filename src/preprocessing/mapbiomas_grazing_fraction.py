"""Fraction of grazing-compatible land on the ERA5-Land 0,1 degree grid.

MapBiomas Colombia Collection 3 (30 m) classes, identified by matching raster
areas with the official national statistics (2024):
  21 = Mosaico de agricultura o pasto, 12 = Formacion herbacea (natural savanna),
  11 = Formacion natural no forestal inundable (flooded savanna).
Grazing-compatible land = {21, 12, 11}; the narrower definition {21} is kept for
sensitivity. Fractions are area-weighted (pixel area varies with latitude) and
computed over observed pixels (value 0 = outside the mapped area).
Output: data/interim/mapbiomas/grazing_fraction_0p1.nc
"""
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mapbiomas"
OUT = ROOT / "data" / "interim" / "mapbiomas" / "grazing_fraction_0p1.nc"
YEARS = [2017, 2020, 2024]
GRAZING, MOSAIC = (21, 12, 11), (21,)
# 0,1 degree cells centred on multiples of 0,1 (ERA5-Land grid); edges at +-0,05
LAT_TOP, LON_LEFT, STEP = 13.65, -81.85, 0.1
NLAT, NLON = 180, 151  # centres 13.6 ... -4.3 and -81.8 ... -66.8
R_EARTH = 6371008.8


def aggregate(path: Path) -> dict[str, np.ndarray]:
    acc = {k: np.zeros((NLAT, NLON)) for k in ("valid", "grazing", "mosaic")}
    with rasterio.open(path) as src:
        t = src.transform
        dx = np.deg2rad(abs(t.a)) * R_EARTH
        dy = np.deg2rad(abs(t.e)) * R_EARTH
        for _, win in src.block_windows(1):
            a = src.read(1, window=win)
            rows = np.arange(win.row_off, win.row_off + win.height)
            cols = np.arange(win.col_off, win.col_off + win.width)
            lat = t.f + (rows + 0.5) * t.e
            lon = t.c + (cols + 0.5) * t.a
            iy = np.floor((LAT_TOP - lat) / STEP).astype(int)
            ix = np.floor((lon - LON_LEFT) / STEP).astype(int)
            okr, okc = (iy >= 0) & (iy < NLAT), (ix >= 0) & (ix < NLON)
            if not okr.any() or not okc.any():
                continue
            a, iy, ix, lat = a[okr][:, okc], iy[okr], ix[okc], lat[okr]
            px = (dx * dy * np.cos(np.deg2rad(lat)))[:, None] / 1e6  # km2 per pixel
            w = np.broadcast_to(px, a.shape)
            II, JJ = np.meshgrid(iy, ix, indexing="ij")
            for key, mask in (("valid", a > 0), ("grazing", np.isin(a, GRAZING)),
                              ("mosaic", np.isin(a, MOSAIC))):
                np.add.at(acc[key], (II[mask], JJ[mask]), w[mask])
    return acc


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lat = np.round(LAT_TOP - STEP / 2 - STEP * np.arange(NLAT), 2)
    lon = np.round(LON_LEFT + STEP / 2 + STEP * np.arange(NLON), 2)
    data = {}
    for y in YEARS:
        acc = aggregate(RAW / f"coverage-colombia_classification_{y}.tif")
        with np.errstate(invalid="ignore", divide="ignore"):
            data[y] = {k: np.where(acc["valid"] > 0, acc[k] / acc["valid"], np.nan)
                       for k in ("grazing", "mosaic")}
        data[y]["valid_km2"] = acc["valid"]
        print(y, "grazing km2:", round(float(acc["grazing"].sum())),
              "mosaic km2:", round(float(acc["mosaic"].sum())), flush=True)
    ds = xr.Dataset(
        {f"{k}_fraction": (("year", "latitude", "longitude"), np.stack([data[y][k] for y in YEARS]))
         for k in ("grazing", "mosaic")},
        coords={"year": YEARS, "latitude": lat, "longitude": lon})
    ds["valid_km2"] = (("year", "latitude", "longitude"), np.stack([data[y]["valid_km2"] for y in YEARS]))
    ds["grazing_fraction_mean"] = ds["grazing_fraction"].mean("year")
    ds["mosaic_fraction_mean"] = ds["mosaic_fraction"].mean("year")
    ds.attrs.update(source="MapBiomas Colombia Collection 3 (coverage, 30 m)",
                    classes_grazing="21,12,11", classes_mosaic="21", years=str(YEARS))
    ds.to_netcdf(OUT)


if __name__ == "__main__":
    main()
