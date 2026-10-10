"""Generic logged download of a single public file into data/raw (immutable).

Usage: python src/download/fetch.py <url> <relative/output/path> <log_name>
Appends url, access time, size and SHA-256 to logs/<log_name>.csv.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
HEADERS = {"User-Agent": "Mozilla/5.0 (research data download; CattleHeatCO)"}


def fetch(url: str, out_rel: str, log_name: str) -> Path:
    out = ROOT / out_rel
    if out.exists():
        raise FileExistsError(f"{out} already exists; raw data are immutable")
    out.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    size = 0
    with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        with out.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                sha.update(chunk)
                size += len(chunk)
    log = ROOT / "logs" / f"{log_name}.csv"
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["url", "accessed_utc", "content_type", "bytes", "sha256", "file"])
        w.writerow([url, datetime.now(timezone.utc).isoformat(timespec="seconds"), ctype, size,
                    sha.hexdigest(), out.relative_to(ROOT).as_posix()])
    print(out, size, sha.hexdigest())
    return out


if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2], sys.argv[3])
