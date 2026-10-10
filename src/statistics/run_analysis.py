"""Main analysis: trends, spatial clusters, decomposition, compound months, sensitivity.

Inputs : data/processed/heat_mun_annual.csv, heat_mun_monthly.csv,
         ica_cattle_analysis.csv, chirps_spi_mun_monthly.csv, dem_mun.csv,
         data/raw/noaa/oni.ascii.txt, MGN 2024, configs/regions.csv
Outputs: results/tables/*.csv and results/analysis_numbers.json (every number
         reported is read from this file).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN, Queen

sys.path.insert(0, str(Path(__file__).parent))
import core  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
PROC, TAB = ROOT / "data/processed", ROOT / "results/tables"
TAB.mkdir(parents=True, exist_ok=True)
CENSUS = list(range(2017, 2026))
P0, P1 = [2017, 2018, 2019], [2023, 2024, 2025]
BASE = (1991, 2020)
N = {}  # numbers reported by the analysis


def load():
    heat = pd.read_csv(PROC / "heat_mun_annual.csv", dtype={"code": str})
    hm = pd.read_csv(PROC / "heat_mun_monthly.csv", dtype={"code": str})
    cat = pd.read_csv(PROC / "ica_cattle_analysis.csv", dtype={"code": str, "dcode": str})
    spi = pd.read_csv(PROC / "chirps_spi_mun_monthly.csv", dtype={"code": str})
    reg = pd.read_csv(ROOT / "configs/regions.csv", dtype=str)[["dcode", "region"]]
    mun = gpd.read_file(f"zip://{ROOT / 'data/raw/dane/MGN2024_MPIO_POLITICO.zip'}!MGN_ADM_MPIO_GRAFICO.shp")
    mun = mun.rename(columns={"mpio_cdpmp": "code", "mpio_narea": "area_km2"})[["code", "area_km2", "geometry"]]
    return heat, hm, cat, spi, reg, mun


def oni_annual() -> pd.Series:
    o = pd.read_csv(ROOT / "data/raw/noaa/oni.ascii.txt", sep=r"\s+")
    return o.groupby("YR")["ANOM"].mean()


def weights_fixed(cat: pd.DataFrame, codes: pd.Index, col: str = "cattle", years=CENSUS) -> pd.Series:
    c = cat[cat.year.isin(years) & cat.code.isin(codes)].pivot_table(index="code", columns="year", values=col)
    w = (c / c.sum()).mean(axis=1)
    return (w / w.sum()).reindex(codes).fillna(0.0)


def national(series: pd.DataFrame, w: pd.Series) -> pd.Series:
    """series: index code, columns year -> herd-weighted national mean per year."""
    s = series.reindex(w.index)
    return (s.mul(w, axis=0)).sum() / w[s.notna().all(axis=1)].sum()


def trend_row(name: str, y: pd.Series, per: int = 10) -> dict:
    t = core.trend_test(y.values)
    lo, hi = core.block_bootstrap_sen_ci(y.values)
    return {"series": name, "start": int(y.index.min()), "end": int(y.index.max()), "test": t["test"],
            "tau": t["tau"], "p": t["p"], "sen_per_decade": t["slope"] * per,
            "ci_low": lo * per, "ci_high": hi * per, "mean_first5": y.iloc[:5].mean(),
            "mean_last5": y.iloc[-5:].mean(), "lag1_r": t["lag1_r"]}


def main() -> None:
    heat, hm, cat, spi, reg, mun = load()
    g = heat[heat.weighting == "grazing"]
    codes = pd.Index(sorted(g.code.unique()), name="code")  # municipalities with ERA5-Land land cells
    N["n_municipalities_climate"] = len(codes)
    N["n_municipalities_total"] = int(mun.code.nunique())
    N["herd_share_without_climate"] = float(
        cat[cat.year.isin(CENSUS) & ~cat.code.isin(codes)].cattle.sum() / cat[cat.year.isin(CENSUS)].cattle.sum())
    piv = lambda d, v: d.pivot_table(index="code", columns="year", values=v)
    DH75 = piv(g, "DH_75")
    w_bar = weights_fixed(cat, codes)
    years_all = sorted(DH75.columns)

    # ---------- H1 national series and secondary outcomes (fixed herd) ----------
    rows, natser = [], {}
    metrics = {"DH_75": "Degree-hours above THI 75", "N_75": "Hours THI>=75", "DH_68.8": "Degree-hours above 68,8",
               "DH_72": "Degree-hours above 72", "DH_79": "Degree-hours above 79", "N_79": "Hours THI>=79",
               "days_dmax_ge79": "Days with daily max THI>=79", "maxrun_ge75_h": "Longest run THI>=75 (h)",
               "thi_p99": "Annual P99 of hourly THI", "thi_dmax_max": "Annual max of daily max THI",
               "thi_mean": "Annual mean THI"}
    for m, lab in metrics.items():
        s = national(piv(g, m), w_bar)
        natser[m] = s
        rows.append(trend_row(m, s))
    trends = pd.DataFrame(rows)
    trends.to_csv(TAB / "trends_national_fixed_herd.csv", index=False)
    pd.DataFrame(natser).to_csv(TAB / "national_series_fixed_herd.csv")
    N["trend_DH75"] = trends.set_index("series").loc["DH_75"].to_dict()
    N["trend_secondary"] = trends.set_index("series").to_dict(orient="index")
    N["n_tests_national"] = len(trends)

    # Realized exposure 2017-2025 and total burden
    C = piv(cat[cat.code.isin(codes)], "cattle").reindex(codes)
    Wy = C / C.sum()
    real = (Wy[CENSUS] * DH75[CENSUS].reindex(codes)).sum()
    burden = (C[CENSUS] * DH75[CENSUS].reindex(codes)).sum()
    exposed79 = {y: float(C.loc[piv(g, "days_dmax_ge79")[y].reindex(codes) >= 30, y].sum() / C[y].sum()) for y in CENSUS}
    pd.DataFrame({"realized_DHbar": real, "burden_animal_degree_hours": burden,
                  "fixed_herd_DHbar": natser["DH_75"][CENSUS], "herd_share_ge30days_danger": pd.Series(exposed79),
                  "cattle": C[CENSUS].sum()}).to_csv(TAB / "realized_exposure_2017_2025.csv")
    N["realized"] = {"DHbar_2017": float(real[2017]), "DHbar_2025": float(real[2025]),
                     "burden_2017": float(burden[2017]), "burden_2025": float(burden[2025]),
                     "share_herd_ge30d_danger_mean": float(np.mean(list(exposed79.values()))),
                     "share_herd_ge30d_danger_by_year": exposed79}

    # ---------- H2 municipal trends ----------
    mt = []
    for c in codes:
        y = DH75.loc[c, years_all]
        t = core.trend_test(y.values)
        mt.append({"code": c, "p": t["p"], "sen_per_decade": t["slope"] * 10, "test": t["test"],
                   "dh75_clim": DH75.loc[c, [yy for yy in years_all if BASE[0] <= yy <= BASE[1]]].mean()})
    mt = pd.DataFrame(mt).set_index("code")
    active = mt.dh75_clim > 0  # municipalities where the metric can change
    mt["sig"] = False
    mt.loc[active, "sig"] = core.bh_fdr(mt.loc[active, "p"].values)
    mt["sig_up"], mt["sig_down"] = mt.sig & (mt.sen_per_decade > 0), mt.sig & (mt.sen_per_decade < 0)
    mt["w_bar"] = w_bar
    mt.to_csv(TAB / "trends_municipal_DH75.csv")
    N["municipal_trends"] = {"n_tested": int(active.sum()), "n_never_exceed": int((~active).sum()),
                             "n_sig_up": int(mt.sig_up.sum()), "n_sig_down": int(mt.sig_down.sum()),
                             "herd_share_sig_up": float(mt.loc[mt.sig_up, "w_bar"].sum()),
                             "herd_share_sig_down": float(mt.loc[mt.sig_down, "w_bar"].sum()),
                             "median_slope_active": float(mt.loc[active, "sen_per_decade"].median())}

    # ---------- spatial weights ----------
    gm = mun.set_index("code").loc[codes].rename_axis("code").reset_index()
    wq = Queen.from_dataframe(gm, use_index=False, silence_warnings=True)
    if wq.islands:
        knn1 = KNN.from_dataframe(gm.to_crs(9377).assign(geometry=lambda d: d.centroid), k=1)
        for i in wq.islands:
            j = knn1.neighbors[i][0]
            wq.neighbors[i], wq.weights[i] = [j], [1.0]
            wq.neighbors[j] = list(wq.neighbors[j]) + [i]
            wq.weights[j] = list(wq.weights[j]) + [1.0]
        wq = type(wq)(wq.neighbors, wq.weights)
    wq.transform = "r"
    wk = KNN.from_dataframe(gm.to_crs(9377).assign(geometry=lambda d: d.centroid), k=6)
    wk.transform = "r"
    np.random.seed(core.SEED)

    def moran(x, w):
        m = Moran(np.asarray(x, float), w, permutations=999)
        return {"I": float(m.I), "p_sim": float(m.p_sim), "z": float(m.z_sim)}

    area = gm.set_index("code")["area_km2"].reindex(codes)
    E = (C[CENSUS].div(area, axis=0) * DH75[CENSUS].reindex(codes))
    logE = np.log1p(E)
    N["moran"] = {"DH75_climatology": moran(mt.dh75_clim.reindex(codes), wq),
                  "sen_slopes": moran(mt.sen_per_decade.reindex(codes), wq),
                  "logE_2025": moran(logE[2025], wq), "logE_2017": moran(logE[2017], wq),
                  "sen_slopes_knn6": moran(mt.sen_per_decade.reindex(codes), wk),
                  "logE_2025_knn6": moran(logE[2025], wk)}

    # ---------- H3 LISA per year ----------
    def lisa_classes(w):
        cls = {}
        for y in CENSUS:
            lm = Moran_Local(logE[y].values, w, permutations=999, seed=core.SEED)
            sig = core.bh_fdr(lm.p_sim)
            cls[y] = np.where(sig, lm.q, 0)  # 1 HH, 2 LH, 3 LL, 4 HL
        return pd.DataFrame(cls, index=codes)

    L = lisa_classes(wq)
    hh = (L == 1)
    persistent = hh.sum(axis=1) >= 7  # >=80 % of the 9 census years
    emerging = (~hh[P0].any(axis=1)) & (hh[P1].sum(axis=1) >= 2)
    L.assign(n_hh=hh.sum(axis=1), persistent=persistent, emerging=emerging).to_csv(TAB / "lisa_classes.csv")
    Lk = lisa_classes(wk)
    pers_k = (Lk == 1).sum(axis=1) >= 7
    jacc = lambda a, b: float((a & b).sum() / max((a | b).sum(), 1))
    share = lambda mask, y=2025: float(C.loc[mask[mask].index, y].sum() / C[y].sum())
    bshare = lambda mask, y=2025: float(burden_mun(mask, y) / (C[y] * DH75[y].reindex(codes)).sum())

    def burden_mun(mask, y):
        idx = mask[mask].index
        return (C.loc[idx, y] * DH75.loc[idx, y]).sum()

    N["lisa"] = {"n_persistent_HH": int(persistent.sum()), "n_emerging_HH": int(emerging.sum()),
                 "herd_share_persistent_2025": share(persistent), "burden_share_persistent_2025": bshare(persistent),
                 "area_share_persistent": float(area[persistent].sum() / area.sum()),
                 "herd_share_emerging_2025": share(emerging),
                 "n_LL_2025": int((L[2025] == 3).sum()), "n_HL_2025": int((L[2025] == 4).sum()),
                 "n_LH_2025": int((L[2025] == 2).sum()), "n_HH_2025": int((L[2025] == 1).sum()),
                 "jaccard_persistent_queen_vs_knn6": jacc(persistent, pers_k),
                 "n_tests_lisa": int(len(codes) * len(CENSUS))}
    rg = pd.Series(reg.set_index("dcode")["region"]).reindex(pd.Index(codes).str[:2]).values
    rg_s = pd.Series(rg, index=codes)
    N["lisa"]["persistent_by_region"] = pd.Series(rg[persistent.values]).value_counts().to_dict()
    N["lisa"]["emerging_by_region"] = pd.Series(rg[emerging.values]).value_counts().to_dict()

    # ---------- H4 decomposition ----------
    def decompose(Cmat, p0, p1, dh=DH75):
        w0 = (Cmat[p0] / Cmat[p0].sum()).mean(axis=1)
        w1 = (Cmat[p1] / Cmat[p1].sum()).mean(axis=1)
        d0, d1 = dh[p0].reindex(Cmat.index).mean(axis=1), dh[p1].reindex(Cmat.index).mean(axis=1)
        red = (w1 - w0) * (d0 + d1) / 2
        cli = (w0 + w1) / 2 * (d1 - d0)
        normal = dh[[y for y in dh.columns if BASE[0] <= y <= BASE[1]]].reindex(Cmat.index).mean(axis=1)
        return {"DHbar_P0": float((w0 * d0).sum()), "DHbar_P1": float((w1 * d1).sum()),
                "delta": float((w1 * d1).sum() - (w0 * d0).sum()), "redistribution": float(red.sum()),
                "climate": float(cli.sum()), "redistribution_normal_climate": float(((w1 - w0) * normal).sum()),
                "by_region": pd.DataFrame({"redistribution": red, "climate": cli,
                                            "region": rg_s.reindex(Cmat.index)}).groupby("region").sum().to_dict()}

    dec = {"primary": decompose(C, P0, P1),
           "S8_P0_2018_2020_P1_2021_2023": decompose(C, [2018, 2019, 2020], [2021, 2022, 2023]),
           "S9_exclude_2024_2025": decompose(C, P0, [2021, 2022, 2023])}
    strict = piv(cat[cat.code.isin(codes)], "cattle_strict").reindex(codes)
    flagged_d = cat[cat.flag_misaligned & cat.year.isin(P0 + P1)].dcode.unique()
    keep = ~pd.Index(codes).str[:2].isin(flagged_d)
    dec["S9_strict_drop_flagged_departments"] = decompose(strict[keep], P0, P1)
    N["decomposition"] = dec
    json.dump(dec, open(TAB / "decomposition.json", "w"), indent=2, default=float)

    # ---------- H6 compound hot-dry months ----------
    mm = hm.merge(spi, on=["code", "year", "month"], how="left")
    mm = mm[mm.code.isin(codes)]
    base = mm[mm.year.between(*BASE)]
    p90 = base.groupby(["code", "month"]).thi_dmax_mean.quantile(0.9).rename("p90")
    mm = mm.merge(p90.reset_index(), on=["code", "month"])
    mm["hot"] = (mm.thi_dmax_mean > mm.p90) & mm.any_ge75.astype(bool)
    comp = {}
    for lab, col, thr in (("spi3_m1", "spi3", -1.0), ("spi6_m1", "spi6", -1.0), ("spi3_m15", "spi3", -1.5)):
        mm["dry"] = mm[col] <= thr
        mm["compound"] = mm.hot & mm.dry
        cy = mm.pivot_table(index="code", columns="year", values="compound", aggfunc="sum").reindex(codes)
        hy = mm.pivot_table(index="code", columns="year", values="hot", aggfunc="sum").reindex(codes)
        s = national(cy, w_bar)
        comp[lab] = {"trend": trend_row(f"compound_{lab}", s), "series": s.to_dict(),
                     "hot_series_trend": trend_row(f"hot_{lab}", national(hy, w_bar)),
                     "cond_prob_dry_given_hot": float(mm.loc[mm.hot, "dry"].mean()),
                     "uncond_prob_dry": float(mm["dry"].mean())}
        if lab == "spi3_m1":
            oni = oni_annual().reindex(s.index)
            nino = (oni >= 0.5).values
            diff, pperm = core.permutation_diff(s.values, nino)
            comp[lab]["enso"] = {"n_nino_years": int(nino.sum()), "mean_nino": float(s[nino].mean()),
                                 "mean_other": float(s[~nino].mean()), "diff": diff, "p_perm": pperm,
                                 "nino_years": [int(v) for v in s.index[nino]]}
            freq = cy.loc[:, cy.columns >= 1991].mean(axis=1)
            dclim = mt.dh75_clim.reindex(codes)
            top_c, top_h = freq >= freq.quantile(0.9), dclim >= dclim.quantile(0.9)
            comp[lab]["top_decile"] = {"jaccard": jacc(top_c, top_h),
                                       "herd_share_compound_only": float(w_bar[top_c & ~top_h].sum()),
                                       "herd_share_heat_top": float(w_bar[top_h].sum()),
                                       "herd_share_compound_top": float(w_bar[top_c].sum())}
            cy.to_csv(TAB / "compound_months_municipal.csv")
            mm[["code", "year", "month", "hot", "dry", "compound"]].to_csv(PROC / "compound_monthly_flags.csv", index=False)
    N["compound"] = comp

    # ---------- H5 metric ranking agreement ----------
    clim = lambda m: piv(g[g.year.between(*BASE)], m).mean(axis=1).reindex(codes)
    tm = clim("thi_mean")
    h5 = {}
    for m in ("DH_75", "maxrun_ge75_h", "days_dmax_ge79", "thi_p99"):
        x = clim(m)
        h5[m] = {"spearman_vs_thi_mean": float(tm.corr(x, method="spearman")),
                 "jaccard_top_decile": jacc(tm >= tm.quantile(0.9), x >= x.quantile(0.9))}
    N["h5_rank_agreement"] = h5

    # ---------- hotspot context profile (descriptive) ----------
    dem = pd.read_csv(PROC / "dem_mun.csv", dtype={"code": str}).set_index("code").reindex(codes)
    wts = pd.read_csv(ROOT / "results/quality_control/weights_qc.csv", dtype={"code": str}).set_index("code").reindex(codes)
    pr = spi[spi.code.isin(codes) & spi.year.between(*BASE)].groupby(["code", "month"]).precip_mm.mean().unstack()
    prof = pd.DataFrame({
        "Elevation (m)": dem["elev_grazing_mean_m"], "Grazing-compatible land (%)": 100 * wts["grazing_share"],
        "Annual precipitation (mm)": pr.sum(axis=1).reindex(codes),
        "Precipitation seasonality (CV, %)": (100 * pr.std(axis=1, ddof=0) / pr.mean(axis=1)).reindex(codes),
        "DH75 1991-2020 (THI-h)": mt.dh75_clim.reindex(codes), "Mean THI 1991-2020": tm,
        "Cattle density 2025 (head/km2)": (C[2025] / area), "Compound months 1991-2025 (per year)":
            mm.pivot_table(index="code", columns="year", values="compound", aggfunc="sum").reindex(codes).loc[:, lambda d: d.columns >= 1991].mean(axis=1)})
    rows = []
    for col in prof:
        a, b = prof.loc[persistent, col], prof.loc[~persistent, col]
        rows.append({"variable": col, "hotspot_median": a.median(), "hotspot_q1": a.quantile(.25),
                     "hotspot_q3": a.quantile(.75), "rest_median": b.median(), "rest_q1": b.quantile(.25),
                     "rest_q3": b.quantile(.75), "cliffs_delta": core.cliffs_delta(a.values, b.values)})
    pd.DataFrame(rows).to_csv(TAB / "hotspot_profile.csv", index=False)
    prof.assign(persistent=persistent, emerging=emerging).to_csv(TAB / "municipal_profile.csv")

    # ---------- sensitivity table ----------
    sens = []
    def add(label, s):
        r = trend_row(label, s)
        sens.append({k: r[k] for k in ("series", "start", "end", "sen_per_decade", "ci_low", "ci_high", "p")})
    for m in ("DH_68.8", "DH_72", "DH_79", "N_75"):
        add(f"S1 {m}", natser[m])
    for wt in ("area", "mosaic"):
        add(f"S2 weighting={wt}", national(piv(heat[heat.weighting == wt], "DH_75"), w_bar))
    add("S3 herd=adults>2y", national(DH75, weights_fixed(cat, codes, "adults_gt2")))
    add("S3 herd=strict", national(DH75, weights_fixed(cat.dropna(subset=["cattle_strict"]), codes, "cattle_strict")))
    add("S4 herd weights 2018-2025", national(DH75, weights_fixed(cat, codes, years=list(range(2018, 2026)))))
    for y0 in (1991, 2001):
        add(f"S5 window {y0}-2025", natser["DH_75"].loc[y0:])
    pd.DataFrame(sens).to_csv(TAB / "sensitivity_trends.csv", index=False)
    N["sensitivity"] = sens

    json.dump(N, open(TAB.parent / "analysis_numbers.json", "w"), indent=2, default=lambda o: o if not isinstance(o, (np.floating, np.integer, np.bool_)) else o.item())
    print(json.dumps({k: N[k] for k in ("trend_DH75", "municipal_trends", "lisa")}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
