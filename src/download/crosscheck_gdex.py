"""Cross-check of the two ERA5-Land access routes on a month downloaded from both.

Builds 2024-03 (the CDS pilot month, kept on disk) from NCAR GDEX (download_era5land_gdex.build_month) and compares it with
the CDS monthly file: time axis, grid, land mask, max |difference| of t2m and d2m, of THI,
and agreement of the THI >= 75 classification.
Output: results/quality_control/gdex_crosscheck.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/download"), str(ROOT / "src/climate")]
import download_era5land_gdex as g  # noqa: E402
from process_era5land import thi_from  # noqa: E402


def main(year: int = 2024, month: int = 3) -> None:
    ds, used = g.build_month(year, month)
    c = xr.open_dataset(ROOT / f"data/raw/era5land/main/era5land_main_{year}{month:02d}.nc")
    out = {"month": f"{year}-{month:02d}", "n_source_files": len(used), "hours": int(ds.sizes["valid_time"]),
           "same_time_axis": bool((ds.valid_time.values == c.valid_time.values).all()),
           "same_grid": bool((ds.latitude.values == np.round(c.latitude.values, 1)).all()
                             and (ds.longitude.values == np.round(c.longitude.values, 1)).all())}
    for v in ("t2m", "d2m"):
        a, b = ds[v].values, c[v].values
        m = np.isfinite(a) & np.isfinite(b)
        out[f"{v}_same_nan_mask"] = bool((np.isnan(a) == np.isnan(b)).all())
        out[f"{v}_max_abs_diff_K"] = float(np.abs(a[m] - b[m]).max())
    ta, _ = thi_from(ds.t2m.values, ds.d2m.values)
    tb, _ = thi_from(c.t2m.values, c.d2m.values)
    m = np.isfinite(ta) & np.isfinite(tb)
    out["thi_max_abs_diff"] = float(np.abs(ta[m] - tb[m]).max())
    out["thi75_classification_agreement"] = float(((ta[m] >= 75) == (tb[m] >= 75)).mean())
    out["n_cell_hours"] = int(m.sum())
    (ROOT / "results/quality_control/gdex_crosscheck.json").write_text(json.dumps(out, indent=2), encoding="utf8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
