"""Hourly ERA5-Land 2 m temperature and dew point for Colombia from the NCAR GDEX mirror.

GDEX dataset d633008 (ERA5-Land hourly data from 1950 to present) serves the ECMWF fields
as global 5-day NetCDF files per variable through THREDDS/OPeNDAP, so the Colombian window
is read directly, without the CDS request queue. The files are stored with bit-grooming
quantization (7 significant digits): against the CDS files the largest difference found
was 0,00024 K and the land-sea mask is identical (test of 2003-01-01..06).

Output is identical in layout to download_era5land.py: one file per month
data/raw/era5land/main/era5land_main_YYYYMM.nc on the same 0,1 degree grid (latitude
13,6 to -4,3, longitude -79,2 to -66,8), hours 00-23 local UTC, same variable names and
encoding, logged to logs/download_era5land.csv with dataset 'gdex-d633008'; the source
files of each month go to logs/download_era5land_gdex_sources.csv.
GDEX files run from hour 01 of day d to hour 00 of day d+5, so hour 00 of the first day
of a month comes from the last file of the previous month.

Usage: python src/download/download_era5land_gdex.py 2004-01 2026-01 [--workers 4] [--reverse]
Workers are separate processes.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import download_era5land as d  # noqa: E402

TDS = "https://tds.gdex.ucar.edu/thredds"
COLL = "files/d633008/e5land.oper.fc.sfc.instan"
VARS = {"t2m": "t2m_167", "d2m": "d2m_168"}
LAT = np.round(np.arange(13.6, -4.3 - 1e-9, -0.1), 1)   # same grid as the CDS files (180)
LON = np.round(np.arange(-79.2, -66.8 + 1e-9, 0.1), 1)  # (125)
SRC_LOG = d.ROOT / "logs" / "download_era5land_gdex_sources.csv"


def month_files(ym: str) -> list[str]:
    xml = requests.get(f"{TDS}/catalog/{COLL}/{ym}/catalog.xml", timeout=120).text
    return sorted(set(re.findall(r'name="(e5land\.oper\.fc\.sfc\.instan\.[^"]+\.nc)"', xml)))


def read_var(fname: str, ym_dir: str, key: str) -> xr.DataArray:
    url = f"{TDS}/dodsC/{COLL}/{ym_dir}/{fname}"
    for attempt in range(6):
        try:
            with xr.open_dataset(url) as ds:
                da = ds[key].sel(latitude=slice(13.65, -4.35), longitude=slice(280.75, 293.25)).load()
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(30 * (attempt + 1))
    da = da.assign_coords(latitude=np.round(da.latitude.values, 1), longitude=np.round(da.longitude.values - 360, 1))
    return da.sel(latitude=LAT, longitude=LON)


def build_month(year: int, month: int) -> tuple[xr.Dataset, list[str]]:
    p = pd.Period(f"{year}-{month:02d}")
    prev = (p - 1).strftime("%Y%m")
    cur = p.strftime("%Y%m")
    t0, t1 = p.start_time, p.end_time.floor("h")
    out, used = {}, []
    files = {cur: month_files(cur), prev: month_files(prev)}
    for key, tag in VARS.items():
        parts = []
        for ym_dir in (prev, cur):
            for f in files[ym_dir]:
                if f".{tag}." not in f:
                    continue
                a, b = re.search(r"\.(\d{10})-(\d{10})\.nc$", f).groups()
                fa, fb = pd.Timestamp(a[:8] + "T" + a[8:]), pd.Timestamp(b[:8] + "T" + b[8:])
                if fb < t0 or fa > t1:
                    continue
                parts.append(read_var(f, ym_dir, key))
                used.append(f"{ym_dir}/{f}")
        da = xr.concat(parts, dim="valid_time").sortby("valid_time")
        da = da.sel(valid_time=slice(t0, t1))
        assert da.sizes["valid_time"] == p.days_in_month * 24, f"{cur} {key}: {da.sizes['valid_time']} h"
        assert not da.indexes["valid_time"].duplicated().any(), f"{cur} {key}: duplicated hours"
        out[key] = da.astype("float32")
    ds = xr.Dataset(out)
    ds["t2m"].attrs.update(units="K", long_name="2 metre temperature")
    ds["d2m"].attrs.update(units="K", long_name="2 metre dewpoint temperature")
    ds.attrs.update(source="ERA5-Land hourly, NCAR GDEX d633008 (THREDDS/OPeNDAP subset)",
                    institution="European Centre for Medium-Range Weather Forecasts",
                    history=f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} subset by download_era5land_gdex.py")
    return ds, used


def fetch(year: int, month: int) -> tuple[dict, dict] | None:
    """Runs in a worker process (the netCDF library is not safe for concurrent remote reads
    from threads); returns the log rows, which the parent process writes."""
    if d.done("main", year, month):
        return None
    t0 = datetime.now(timezone.utc)
    ds, used = build_month(year, month)
    out = d.RAW / "main" / f"era5land_main_{year}{month:02d}.nc"
    n = ds.sizes["valid_time"]
    enc = {v: {"zlib": True, "complevel": 1, "shuffle": True, "dtype": "float32", "_FillValue": float("nan"),
               "chunksizes": (min(372, n), 90, 63)} for v in ("t2m", "d2m")}
    enc["valid_time"] = {"units": "seconds since 1970-01-01", "dtype": "int64"}
    tmp = out.with_suffix(".part")
    ds.to_netcdf(tmp, format="NETCDF4", encoding=enc)
    tmp.replace(out)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row = {"dataset": "gdex-d633008", "domain": "main", "year": year, "month": month,
           "requested_utc": t0.isoformat(timespec="seconds"), "completed_utc": now,
           "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "file": out.relative_to(d.ROOT).as_posix()}
    return row, {"year": year, "month": month, "accessed_utc": now, "source_files": " ".join(used)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reverse", action="store_true", help="newest months first")
    a = ap.parse_args()
    jobs = d.months(a.start, a.end)
    if a.reverse:
        jobs = jobs[::-1]
    # a worker sometimes dies inside the netCDF/OPeNDAP C library (BrokenProcessPool, no
    # traceback); the pool is then rebuilt and the pending months are asked again
    for rnd in range(1, 101):
        todo = [j for j in jobs if not d.done("main", *j)]
        if not todo:
            break
        if rnd > 1:
            time.sleep(30)
        print(f"round {rnd}: {len(todo)} months pending", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(fetch, *j): j for j in todo}
            for f in as_completed(futs):
                j = futs[f]
                try:
                    rows = f.result()
                except Exception as e:
                    print("FAILED", j, repr(e)[:200], flush=True)
                    continue
                if rows:
                    d.log_row(rows[0])
                    d.log_row(rows[1], SRC_LOG)
                    print(Path(rows[0]["file"]).name, rows[0]["bytes"], flush=True)


if __name__ == "__main__":
    main()
