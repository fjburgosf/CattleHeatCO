"""Download hourly ERA5-Land 2 m temperature and dew point for Colombia (CDS API).

One NetCDF file per month and domain is written to data/raw/era5land/<domain>/.
Existing files are skipped (raw data are immutable). Each monthly file is logged to
logs/download_era5land.csv with size and SHA-256.

The CDS serves one request per user at a time and each request waits in a shared queue
(30-70 min on 2026-10-06), so with --chunk N up to N months of the same year are asked
in one request (CDS size limit: 12.000 units, one month = 2.976, so N <= 4). The
multi-month file is split into the usual monthly files (values unchanged, same
encoding) and its own SHA-256 is logged to logs/download_era5land_chunks.csv.

Usage:
  python src/download/download_era5land.py 2024-03 2024-03            # pilot month
  python src/download/download_era5land.py 1981-01 2026-01 --workers 3
  python src/download/download_era5land.py 1981-01 2026-01 --workers 2 --chunk 4
Domains: 'main' (mainland Colombia). 'sai' (San Andres and Providencia) was piloted for
2024-03 and contains no ERA5-Land land cell, so the archipelago is excluded.
The extra month 2026-01 is needed to close local-time (UTC-5) year 2025.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "era5land"
LOG = ROOT / "logs" / "download_era5land.csv"
CHUNK_LOG = ROOT / "logs" / "download_era5land_chunks.csv"
DATASET = "reanalysis-era5-land"
VARIABLES = ["2m_temperature", "2m_dewpoint_temperature"]
# area = [North, West, South, East] in degrees
DOMAINS = {"main": [13.6, -79.2, -4.3, -66.8], "sai": [13.5, -81.8, 12.4, -81.3]}
_lock = threading.Lock()


def months(start: str, end: str) -> list[tuple[int, int]]:
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def log_row(row: dict, log: Path = LOG) -> None:
    with _lock:
        new = not log.exists()
        with log.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


def deleted_files() -> set[str]:
    """Months already processed and deleted on purpose after their SHA-256 was checked."""
    f = ROOT / "logs" / "era5land_deleted.csv"
    if not f.exists():
        return set()
    return {r["file"] for r in csv.DictReader(open(f, encoding="utf-8"))}


def done(domain: str, year: int, month: int) -> bool:
    name = f"era5land_{domain}_{year}{month:02d}.nc"
    return (RAW / domain / name).exists() or name in deleted_files()


def fetch(domain: str, year: int, month: int) -> None:
    out = RAW / domain / f"era5land_{domain}_{year}{month:02d}.nc"
    if done(domain, year, month):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    request = {
        "variable": VARIABLES,
        "year": f"{year}",
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": DOMAINS[domain],
    }
    t0 = datetime.now(timezone.utc)
    for attempt in range(200):  # the CDS rejects jobs while the per-user queue is full
        try:
            cdsapi.Client(quiet=True, progress=False).retrieve(DATASET, request, str(tmp))
            break
        except Exception as e:
            if "queued requests" not in str(e) or attempt == 199:
                raise
            time.sleep(180)
    tmp.replace(out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    log_row({"dataset": DATASET, "domain": domain, "year": year, "month": month,
             "requested_utc": t0.isoformat(timespec="seconds"),
             "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "bytes": out.stat().st_size, "sha256": sha, "file": out.relative_to(ROOT).as_posix()})
    print(out.name, out.stat().st_size, flush=True)


def retrieve(request: dict, target: Path) -> None:
    for attempt in range(200):  # the CDS rejects jobs while the per-user queue is full
        try:
            cdsapi.Client(quiet=True, progress=False).retrieve(DATASET, request, str(target))
            return
        except Exception as e:
            if "queued requests" not in str(e) or attempt == 199:
                raise
            time.sleep(180)


def split_months(src: Path, domain: str, year: int, mons: list[int]) -> list[Path]:
    """Write one monthly file per month from a multi-month CDS file (same encoding)."""
    out_files = []
    with xr.open_dataset(src) as ds:
        ds = ds.load()
    for m in mons:
        out = RAW / domain / f"era5land_{domain}_{year}{m:02d}.nc"
        sub = ds.sel(valid_time=(ds["valid_time"].dt.year == year) & (ds["valid_time"].dt.month == m))
        n_exp = pd.Period(f"{year}-{m:02d}").days_in_month * 24
        assert sub.sizes["valid_time"] == n_exp, f"{out.name}: {sub.sizes['valid_time']} h, expected {n_exp}"
        enc = {v: {"zlib": True, "complevel": 1, "shuffle": True, "dtype": "float32", "_FillValue": float("nan"),
                   "chunksizes": (min(372, n_exp), 90, 63)} for v in ("t2m", "d2m")}
        enc["valid_time"] = {"units": "seconds since 1970-01-01", "dtype": "int64"}
        tmp = out.with_suffix(".part")
        sub.to_netcdf(tmp, format="NETCDF4", encoding=enc)
        tmp.replace(out)
        out_files.append(out)
    return out_files


def fetch_chunk(domain: str, year: int, mons: list[int]) -> None:
    mons = [m for m in mons if not done(domain, year, m)]
    if not mons:
        return
    (RAW / domain).mkdir(parents=True, exist_ok=True)
    tag = "-".join(f"{m:02d}" for m in mons)
    tmp = RAW / domain / f"era5land_{domain}_{year}_{tag}.chunk.part"
    request = {
        "variable": VARIABLES,
        "year": f"{year}",
        "month": [f"{m:02d}" for m in mons],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": DOMAINS[domain],
    }
    t0 = datetime.now(timezone.utc)
    retrieve(request, tmp)
    t1 = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_row({"dataset": DATASET, "domain": domain, "year": year, "months": " ".join(map(str, mons)),
             "requested_utc": t0.isoformat(timespec="seconds"), "completed_utc": t1,
             "bytes": tmp.stat().st_size, "sha256": hashlib.sha256(tmp.read_bytes()).hexdigest()}, CHUNK_LOG)
    for out in split_months(tmp, domain, year, mons):
        log_row({"dataset": DATASET, "domain": domain, "year": year, "month": int(out.stem[-2:]),
                 "requested_utc": t0.isoformat(timespec="seconds"), "completed_utc": t1,
                 "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                 "file": out.relative_to(ROOT).as_posix()})
        print(out.name, out.stat().st_size, flush=True)
    tmp.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--domains", default="main")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=1, help="months per CDS request (same year, <= 4)")
    a = ap.parse_args()
    jobs = [(d, y, m) for (y, m) in months(a.start, a.end) for d in a.domains.split(",")]
    for rnd in range(1, 11):  # repeat until every month exists (CDS returns sporadic 5xx errors)
        todo = [j for j in jobs if not done(*j)]
        if not todo:
            break
        print(f"round {rnd}: {len(todo)} months pending", flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            if a.chunk > 1:
                groups = {}
                for d, y, m in todo:
                    groups.setdefault((d, y), []).append(m)
                tasks = [(d, y, ms[i:i + a.chunk]) for (d, y), ms in groups.items() for i in range(0, len(ms), a.chunk)]
                futs = {ex.submit(fetch_chunk, *t): t for t in tasks}
            else:
                futs = {ex.submit(fetch, *j): j for j in todo}
            for f, j in futs.items():
                try:
                    f.result()
                except Exception as e:  # logged and retried in the next round
                    print("FAILED", j, repr(e)[:200], flush=True)


if __name__ == "__main__":
    main()
