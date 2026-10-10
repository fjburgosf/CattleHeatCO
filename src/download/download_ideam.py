"""IDEAM automatic-station air temperature and relative humidity (datos.gov.co, Socrata).

Datasets: sbwg-7ju4 (Temperatura Ambiente del Aire, 2 m, deg C) and uext-mhny
(Humedad del Aire, 2 m, %). Step 1 queries, per year, the number of observations per
station in each dataset. Step 2 downloads the raw observations of station-years with
at least MIN_OBS observations in both datasets (paged, 50.000 rows per request).
Raw files: data/raw/ideam/obs/<dataset>/<station>_<year>.csv (with the sensor code). Air
humidity is stored under sensor 0027 up to 2023 and under 0027/0028 (GPRS) from 2024; both are
downloaded and the validation chooses per station-year (see validate_ideam.py).
Log: logs/download_ideam.csv
Usage: python src/download/download_ideam.py 2017 2025
"""
from __future__ import annotations

import csv
import hashlib
import io
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/ideam"
LOG = ROOT / "logs/download_ideam.csv"
DATASETS = {"temp": "sbwg-7ju4", "rh": "uext-mhny"}
MIN_OBS = int(0.8 * 8760)  # at least the equivalent of 80 % hourly coverage
PAGE = 50000


def soql(ds: str, params: dict, timeout: int = 900) -> pd.DataFrame:
    url = f"https://www.datos.gov.co/resource/{ds}.csv?" + urllib.parse.urlencode(params, safe="(),'*:")
    for attempt in range(6):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CattleHeatCO"}),
                                         timeout=timeout).read()
            return pd.read_csv(io.BytesIO(raw), dtype=str)
        except Exception as e:  # transient Socrata errors
            if attempt == 5:
                raise
            time.sleep(30 * (attempt + 1))


def coverage(year: int) -> pd.DataFrame:
    """Observations per station and year, queried month by month (a whole-year
    aggregate exceeds the Socrata time limit for the older years)."""
    out = []
    for var, ds in DATASETS.items():
        f = RAW / f"coverage_{var}_{year}.csv"
        if not f.exists():
            parts = []
            for m in range(1, 13):
                mf = RAW / "coverage_months" / f"{var}_{year}{m:02d}.csv"  # resumable cache
                if mf.exists():
                    parts.append(pd.read_csv(mf, dtype=str))
                    continue
                last = pd.Period(f"{year}-{m:02d}").days_in_month
                part = soql(ds, {
                    "$select": "codigoestacion,count(valorobservado) as n,min(fechaobservacion) as t0,"
                               "max(fechaobservacion) as t1,max(latitud) as lat,max(longitud) as lon,"
                               "max(nombreestacion) as nombre",
                    "$where": f"fechaobservacion between '{year}-{m:02d}-01T00:00:00' and "
                              f"'{year}-{m:02d}-{last}T23:59:59'",
                    "$group": "codigoestacion", "$limit": "20000"})
                mf.parent.mkdir(parents=True, exist_ok=True)
                part.to_csv(mf, index=False)
                parts.append(part)
            d = pd.concat(parts)
            d["n"] = pd.to_numeric(d["n"])
            d = d.groupby("codigoestacion").agg(n=("n", "sum"), t0=("t0", "min"), t1=("t1", "max"),
                                                lat=("lat", "max"), lon=("lon", "max"),
                                                nombre=("nombre", "max")).reset_index()
            d.to_csv(f, index=False)
        d = pd.read_csv(f, dtype={"codigoestacion": str})
        d["var"] = var
        out.append(d)
    return pd.concat(out)


def download(var: str, station: str, year: int) -> None:
    f = RAW / "obs" / var / f"{station}_{year}.csv"
    if f.exists():
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    parts, off = [], 0
    while True:
        d = soql(DATASETS[var], {"$select": "codigosensor,fechaobservacion,valorobservado",
                                 "$where": f"codigoestacion='{station}' AND fechaobservacion between "
                                           f"'{year}-01-01T00:00:00' and '{year}-12-31T23:59:59'",
                                 "$order": "fechaobservacion", "$limit": str(PAGE), "$offset": str(off)})
        parts.append(d)
        if len(d) < PAGE:
            break
        off += PAGE
    d = pd.concat(parts)
    tmp = f.with_suffix(".part")
    d.to_csv(tmp, index=False)
    tmp.replace(f)
    b = f.read_bytes()
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["dataset", "station", "year", "rows", "accessed_utc", "sha256", "file"])
        w.writerow([DATASETS[var], station, year, len(d), datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    hashlib.sha256(b).hexdigest(), f.relative_to(ROOT).as_posix()])


def main(y0: int, y1: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for y in range(y1, y0 - 1, -1):  # recent years first (faster queries, more stations)
        cov = coverage(y)
        n = cov.pivot_table(index="codigoestacion", columns="var", values="n", aggfunc="sum").fillna(0)
        sel = n[(n.get("temp", 0) >= MIN_OBS) & (n.get("rh", 0) >= MIN_OBS)].index
        print(y, "stations in both datasets:", len(n.dropna()), "selected:", len(sel), flush=True)
        for st in sel:
            for var in DATASETS:
                download(var, st, y)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
