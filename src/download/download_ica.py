"""Download the ICA municipal bovine census tables (raw, immutable).

Each file is stored under data/raw/ica/ with its original bytes. The URL, access
time, HTTP content type, size and SHA-256 are appended to logs/download_ica.csv.
"""
from __future__ import annotations

import csv
import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ica"
LOG = ROOT / "logs" / "download_ica.csv"
BASE = "https://www.ica.gov.co"

# Links listed on the ICA census index page
# (https://www.ica.gov.co/areas/pecuaria/servicios/epidemiologia-veterinaria/censos-2016/censo-2018),
# read on 2026-10-04. 2017 comes from .../censos-2016/censo-2017. The 2016 entry on the ICA
# site is only a departmental PDF map (no municipal table), so 2016 is not downloaded.
SOURCES = {
    2017: "/getattachment/Areas/Pecuaria/Servicios/Epidemiologia-Veterinaria/Censos-2016/Censo-2017/CENSO-BOVINO-2017.xlsx.aspx?lang=es-CO",
    2018: "/areas/pecuaria/servicios/epidemiologia-veterinaria/censos-2016/censo-2018/censo-bovinos-por-departamento-2018.aspx",
    2019: "/areas/pecuaria/servicios/epidemiologia-veterinaria/censos-2016/censo-2018/tabla-bovinos-municipios-departamentos-2019.aspx",
    2020: "/areas/pecuaria/servicios/epidemiologia-veterinaria/censos-2016/censo-2020/bovinos-censo-2020.aspx",
    2021: "/areas/pecuaria/servicios/epidemiologia-veterinaria/censos-2016/censo-2018/censo-bovino-2021-1.aspx",
    2022: "/areas/pecuaria/servicios/epidemiologia-veterinaria/censos-2016/censo-2018/censos-bovinos-2022.aspx",
    2023: "/areas/pecuaria/servicios/epidemiologia-veterinaria/censos-bovinos-2023-final.aspx",
    2024: "/areas/pecuaria/servicios/enfermedades-animales/fiebre-aftosa/censo-bovino-final.aspx",
    2025: "/areas/pecuaria/servicios/epidemiologia-veterinaria/sensores-epidemiologicos-pecuarios/censo-bovino-2025_14-5-25.aspx",
    2026: "/areas/pecuaria/servicios/epidemiologia-veterinaria/sensores-epidemiologicos-pecuarios/tabla-de-poblacion-bovina-por-municipio-y-por-depa.aspx",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (research data download; CattleHeatCO)"}


def extension(content_type: str, url: str) -> str:
    ct = content_type.split(";")[0].strip().lower()
    known = {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.ms-excel": ".xls",
        "application/pdf": ".pdf",
        "text/html": ".html",
    }
    return known.get(ct) or mimetypes.guess_extension(ct) or Path(url).suffix or ".bin"


def download(year: int, path: str) -> dict:
    url = BASE + path
    r = requests.get(url, headers=HEADERS, timeout=120, allow_redirects=True)
    r.raise_for_status()
    ext = extension(r.headers.get("Content-Type", ""), r.url)
    out = RAW / f"ica_bovinos_{year}{ext}"
    if out.exists():
        raise FileExistsError(f"{out} already exists; raw data are immutable")
    out.write_bytes(r.content)
    return {
        "year": year,
        "url": url,
        "final_url": r.url,
        "accessed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_type": r.headers.get("Content-Type", ""),
        "bytes": len(r.content),
        "sha256": hashlib.sha256(r.content).hexdigest(),
        "file": out.relative_to(ROOT).as_posix(),
    }


def main(years: list[int] | None = None) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new_log = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as fh:
        writer = None
        for year, path in SOURCES.items():
            if years and year not in years:
                continue
            row = download(year, path)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if new_log:
                    writer.writeheader()
            writer.writerow(row)
            print(row["year"], row["content_type"], row["bytes"], row["file"])


if __name__ == "__main__":
    import sys

    main([int(a) for a in sys.argv[1:]] or None)
