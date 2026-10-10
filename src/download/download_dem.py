"""Download Copernicus DEM GLO-90 tiles covering mainland Colombia (public AWS bucket).

Tiles are 1 x 1 degree Cloud-Optimized GeoTIFFs named by their south-west corner.
Ocean tiles do not exist and return 404 (recorded as 'absent').
Source: https://copernicus-dem-90m.s3.amazonaws.com (ESA Copernicus DEM, GLO-90).
"""
from pathlib import Path
import csv
import hashlib
from datetime import datetime, timezone

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "dem_glo90"
LOG = ROOT / "logs" / "download_dem.csv"
BASE = "https://copernicus-dem-90m.s3.amazonaws.com"


def tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"W{-lon:03d}" if lon < 0 else f"E{lon:03d}"
    return f"Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["tile", "status", "accessed_utc", "bytes", "sha256"])
        for lat in range(-5, 14):          # south-west corners -5 ... 13
            for lon in range(-80, -66):    # -80 ... -67
                name = tile_name(lat, lon)
                dst = OUT / f"{name}.tif"
                if dst.exists():
                    continue
                r = requests.get(f"{BASE}/{name}/{name}.tif", timeout=300)
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                if r.status_code == 404:
                    w.writerow([name, "absent", now, 0, ""])
                    continue
                r.raise_for_status()
                dst.write_bytes(r.content)
                w.writerow([name, "ok", now, len(r.content), hashlib.sha256(r.content).hexdigest()])
                fh.flush()


if __name__ == "__main__":
    main()
