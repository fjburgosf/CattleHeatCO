"""Partial validation of hourly THI from ERA5-Land against IDEAM automatic stations
ERA5-Land is not corrected; the comparison characterises uncertainty.

Station data (data/raw/ideam/obs/{temp,rh}/<station>_<year>.csv):
  - sensors: one per variable and station-year, the one with most data (T: 0068/0071;
    RH: 0027, which holds all air humidity up to 2023, or 0028 GPRS from 2024);
  - physical test: within-day anomalies of RH and T must correlate below -0,3 (air humidity
    falls as temperature rises; a stuck or soil-moisture series fails);
  - range checks: -10 <= T <= 50 degC, 1 <= RH <= 100 %; station-years with a stuck
    humidity sensor (SD < 3 %, > 1 % of hours <= 5 % or > 50 % of hours >= 99,5 %) dropped;
  - hourly value = mean of observations within +-30 min of the full hour (ERA5-Land
    values are instantaneous at the full hour);
  - time stamps: the lag that maximises the correlation of the mean diurnal temperature
    cycle with ERA5-Land is computed for every station-year; a year whose median lag is
    between -6 and -4 h is published in UTC (found for 2025) and its stamps are shifted to
    local time (UTC-5) before the comparison; other years are local;
  - a station-year is kept when >= 80 % of its hours have both T and RH.
ERA5-Land: nearest land cell; cell elevation from the GLO-90 grid (dem_grid_0p1.nc);
station elevation from the IDEAM catalogue. Results stratified by |dz| <= 300 m.
Metrics per station-year: bias and RMSE of hourly THI, Pearson r, ratio of hours with
THI >= 75 (ERA5/station) and ratio of degree-hours above 75.
Outputs: results/tables/validation_ideam_station_years.csv, results/validation_ideam.json
Usage: python src/statistics/validate_ideam.py 2017 2025
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
OBS = ROOT / "data/raw/ideam/obs"
ERA = ROOT / "data/raw/era5land/main"
sys.path.insert(0, str(ROOT / "src/climate"))
from process_era5land import thi_from  # noqa: E402

UTC5 = pd.Timedelta(hours=5)


def station_hourly(st: str, year: int, shift_h: int = 0) -> pd.DataFrame | None:
    fs = {v: OBS / v / f"{st}_{year}.csv" for v in ("temp", "rh")}
    if not all(f.exists() for f in fs.values()):
        return None
    out = {}
    for v, f in fs.items():
        d = pd.read_csv(f)
        if d.empty:
            return None
        d["codigosensor"] = d["codigosensor"].astype(str).str.zfill(4)
        # one sensor per variable and station-year (the one with more data): T 0068/0071,
        # RH 0027 (all years) or 0028 (GPRS, from 2024); never average two instruments
        d = d[d["codigosensor"] == d["codigosensor"].value_counts().idxmax()]
        if d.empty:
            return None
        t = pd.to_datetime(d["fechaobservacion"]) + pd.Timedelta(hours=shift_h)
        x = pd.to_numeric(d["valorobservado"], errors="coerce")
        ok = x.between(-10, 50) if v == "temp" else x.between(1, 100)
        hour = (t + pd.Timedelta(minutes=30)).dt.floor("h")  # +-30 min window around the full hour
        out[v] = x[ok].groupby(hour[ok]).mean()
    h = pd.DataFrame(out).dropna()
    # plausibility: a stuck humidity sensor (near-constant or saturated) invalidates the year
    if h.empty or h["rh"].std() < 3 or (h["rh"] <= 5).mean() > 0.01 or (h["rh"] >= 99.5).mean() > 0.5:
        return None
    # physical plausibility of air humidity: within-day RH falls when T rises
    anom = h - h.groupby(h.index.normalize()).transform("mean")
    if anom["temp"].corr(anom["rh"]) > -0.3:
        return None
    full = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    h = h.reindex(full)
    return h


def era_series(lat: float, lon: float, year: int, cache: dict) -> pd.DataFrame | None:
    """Hourly T and Td (K) at the nearest land cell, local time, for one year."""
    key = (round(lat, 1), round(lon, 1), year)
    if key in cache:
        return cache[key]
    months = pd.period_range(f"{year}-01", f"{year + 1}-01", freq="M")
    files = [ERA / f"era5land_main_{p.year}{p.month:02d}.nc" for p in months]
    if not all(f.exists() for f in files):
        return None
    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        tn = "valid_time" if "valid_time" in ds.dims else "time"
        land = ds["t2m"].isel({tn: 0}).notnull()
        cand = land.where(land, drop=True)
        la = cand["latitude"].values[:, None]
        lo = cand["longitude"].values[None, :]
        d2 = (la - lat) ** 2 + (lo - lon) ** 2
        d2 = np.where(cand.values, d2, np.inf)
        i, j = np.unravel_index(np.argmin(d2), d2.shape)
        cell = ds.sel(latitude=float(cand["latitude"][i]), longitude=float(cand["longitude"][j]))
        parts.append(pd.DataFrame({"t2m": cell["t2m"].values, "d2m": cell["d2m"].values,
                                   "clat": float(cand["latitude"][i]), "clon": float(cand["longitude"][j])},
                                  index=pd.DatetimeIndex(cell[tn].values) - UTC5))
    s = pd.concat(parts)
    s = s[s.index.year == year]
    cache[key] = s
    return s


def main(y0: int, y1: int) -> None:
    cat = pd.read_csv(ROOT / "data/raw/ideam/catalogo_estaciones_ideam.csv", dtype={"Codigo": str})
    cat["key"] = cat["Codigo"].str.lstrip("0")
    dem = xr.open_dataset(ROOT / "data/interim/dem_grid_0p1.nc")["elev_m"]
    rows, lags, cache, utc_years = [], [], {}, []

    def best_lag(h, e):
        """Lag (h) that best aligns the mean diurnal temperature cycle with ERA5-Land."""
        j = h.join(e, how="inner")
        dc_s = j.groupby(j.index.hour)["temp"].mean()
        dc_e = (j.groupby(j.index.hour)["t2m"].mean() - 273.15)
        return max(range(-6, 7), key=lambda k: np.corrcoef(np.roll(dc_s.values, k), dc_e.values)[0, 1])

    for year in range(y0, y1 + 1):
        covf = ROOT / f"data/raw/ideam/coverage_temp_{year}.csv"
        if not covf.exists():
            continue
        cov = pd.read_csv(covf, dtype={"codigoestacion": str}).set_index("codigoestacion")
        stations = sorted({p.stem.rsplit("_", 1)[0] for p in (OBS / "temp").glob(f"*_{year}.csv")})
        # pass 1: time-stamp convention of the year from the median diurnal-cycle lag
        year_lags = []
        for st in stations:
            h = station_hourly(st, year)
            e = era_series(float(cov.loc[st, "lat"]), float(cov.loc[st, "lon"]), year, cache) if h is not None else None
            if h is not None and e is not None:
                year_lags.append(best_lag(h, e))
        shift = -5 if year_lags and -6 <= float(np.median(year_lags)) <= -4 else 0
        if shift:
            utc_years.append(year)
        # pass 2: comparison with the stamps in local time
        for st in stations:
            h = station_hourly(st, year, shift)
            if h is None:
                continue
            completeness = float(h.notna().all(axis=1).mean())
            lat, lon = float(cov.loc[st, "lat"]), float(cov.loc[st, "lon"])
            e = era_series(lat, lon, year, cache)
            if e is None:
                continue
            thi_e, _ = thi_from(e["t2m"].values, e["d2m"].values)
            e = e.assign(thi=thi_e)
            j = h.join(e, how="inner")
            lag = best_lag(h, e)
            lags.append(lag)
            if completeness < 0.8:
                continue
            t, rh = j["temp"], j["rh"]
            thi_s = (1.8 * t + 32) - (0.55 - 0.0055 * rh) * (1.8 * t - 26)
            ok = thi_s.notna() & j["thi"].notna()
            a, b = j.loc[ok, "thi"], thi_s[ok]
            info = cat[cat["key"] == st.lstrip("0")]
            z_st = float(info["Altitud"].iloc[0]) if len(info) else np.nan
            z_cell = float(dem.sel(latitude=e["clat"].iloc[0], longitude=e["clon"].iloc[0], method="nearest"))
            rows.append({"station": st, "year": year, "name": cov.loc[st, "nombre"], "lat": lat, "lon": lon,
                         "dept": info["Departamento"].iloc[0] if len(info) else "", "z_station_m": z_st,
                         "z_cell_m": z_cell, "dz_m": z_cell - z_st, "completeness": completeness,
                         "n_hours": int(ok.sum()), "best_lag_h": lag, "utc_shift_h": shift,
                         "bias_thi": float((a - b).mean()), "rmse_thi": float(np.sqrt(((a - b) ** 2).mean())),
                         "r_thi": float(np.corrcoef(a, b)[0, 1]),
                         "bias_t_c": float((j.loc[ok, "t2m"] - 273.15 - t[ok]).mean()),
                         "hours75_station": int((b >= 75).sum()), "hours75_era": int((a >= 75).sum()),
                         "dh75_station": float(np.clip(b - 75, 0, None).sum()),
                         "dh75_era": float(np.clip(a - 75, 0, None).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/tables/validation_ideam_station_years.csv", index=False)
    if df.empty:
        print("no station-year with ERA5-Land data yet"); return
    df["stratum"] = np.where(df["dz_m"].abs() <= 300, "dz<=300m", "dz>300m")

    def summ(d):
        return {"n_station_years": int(len(d)), "n_stations": int(d.station.nunique()),
                "bias_thi_median": float(d.bias_thi.median()), "bias_thi_iqr": [float(d.bias_thi.quantile(.25)), float(d.bias_thi.quantile(.75))],
                "rmse_thi_median": float(d.rmse_thi.median()), "r_thi_median": float(d.r_thi.median()),
                "bias_t_median": float(d.bias_t_c.median()),
                "ratio_hours75_pooled": float(d.hours75_era.sum() / max(d.hours75_station.sum(), 1)),
                "ratio_dh75_pooled": float(d.dh75_era.sum() / max(d.dh75_station.sum(), 1e-9)),
                "spearman_dh75_station_vs_era": float(d[["dh75_station", "dh75_era"]].corr(method="spearman").iloc[0, 1])}
    res = {"all": summ(df), **{s: summ(g) for s, g in df.groupby("stratum")},
           "timezone_lag_mode_h": int(pd.Series(lags).mode().iloc[0]), "timezone_lag_counts": pd.Series(lags).value_counts().to_dict(),
           "years_published_in_utc": utc_years, "station_years_by_year": df.groupby("year").size().to_dict()}
    (ROOT / "results/validation_ideam.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
