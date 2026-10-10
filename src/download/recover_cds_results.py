"""Recover ERA5-Land requests that finished at the CDS while the downloader was not running.

The CDS keeps the result of a finished job for a while. This script lists the user's most
recent successful jobs, downloads each result whose months are not yet on disk, checks
its size against the CDS record and that no hour is empty, splits it into the usual
monthly files and logs it exactly as download_era5land.py does (chunk and monthly SHA-256).
Usage: python src/download/recover_cds_results.py [n_recent_jobs]
"""
from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import download_era5land as d  # noqa: E402


def main(n_jobs: int = 6) -> None:
    cfg = dict(l.split(":", 1) for l in Path(os.path.expanduser("~/.cdsapirc")).read_text().splitlines() if ":" in l)
    base = cfg["url"].strip().replace("/api", "/api/retrieve/v1/jobs")
    H = {"PRIVATE-TOKEN": cfg["key"].strip()}
    jobs = requests.get(f"{base}?limit={n_jobs}&sortby=-created", headers=H, timeout=60).json()["jobs"]
    for x in jobs:
        if x["status"] != "successful":
            continue
        res = requests.get(f"{base}/{x['jobID']}/results", headers=H, timeout=60)
        if res.status_code != 200:  # result expired
            continue
        a = res.json()["asset"]["value"]
        tmp = d.RAW / "main" / f"recovered_{x['jobID'][:8]}.chunk.part"
        with requests.get(a["href"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for b in r.iter_content(1 << 20):
                    fh.write(b)
        raw = tmp.read_bytes()
        assert len(raw) == a["file:size"], f"{tmp.name}: size mismatch"
        with xr.open_dataset(tmp) as ds:
            t = ds["valid_time"].to_index()
            for v in ("t2m", "d2m"):
                assert int(ds[v].isnull().all(axis=(1, 2)).sum()) == 0, f"{tmp.name}: empty hours in {v}"
        year = int(t.year[0])
        assert (t.year == year).all()
        mons = [m for m in sorted(set(int(m) for m in t.month)) if not d.done("main", year, m)]
        if not mons:
            tmp.unlink()
            continue
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        req = x["created"][:19] + "+00:00"
        d.log_row({"dataset": d.DATASET, "domain": "main", "year": year, "months": " ".join(map(str, mons)),
                   "requested_utc": req, "completed_utc": now, "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest()}, d.CHUNK_LOG)
        for out in d.split_months(tmp, "main", year, mons):
            d.log_row({"dataset": d.DATASET, "domain": "main", "year": year, "month": int(out.stem[-2:]),
                       "requested_utc": req, "completed_utc": now, "bytes": out.stat().st_size,
                       "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                       "file": out.relative_to(d.ROOT).as_posix()})
            print("recovered", out.name, flush=True)
        tmp.unlink()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
