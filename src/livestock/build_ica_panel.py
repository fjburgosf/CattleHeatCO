"""Harmonize the ICA municipal bovine census (2017-2026) into a long panel.

Input : data/raw/ica/*.xls[x] (2019 read from the decrypted copy in data/interim/ica)
        data/raw/dane/MGN2024_MPIO_POLITICO.zip (DIVIPOLA reference)
        configs/ica_name_crosswalk.csv (manual name -> code fixes, documented)
Output: data/processed/ica_bovine_panel.csv
        results/quality_control/ica_*.csv
"""
from __future__ import annotations

import difflib
import re
import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import msoffcrypto
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim" / "ica"
QC = ROOT / "results" / "quality_control"
OUT = ROOT / "data" / "processed" / "ica_bovine_panel.csv"
XWALK = ROOT / "configs" / "ica_name_crosswalk.csv"

FILES = {
    2017: (RAW / "ica" / "ica_bovinos_2017.xlsx", "Bovinos por Muni"),
    2018: (RAW / "ica" / "ica_bovinos_2018.xlsx", "Bovinos por municipio"),
    2019: (INTERIM / "ica_bovinos_2019_decrypted.xls", 0),
    2020: (RAW / "ica" / "ica_bovinos_2020.xlsx", "BOVINOS Y PREDIOS"),
    2021: (RAW / "ica" / "ica_bovinos_2021.xlsx", "BOVINOS Y PREDIOS"),
    2022: (RAW / "ica" / "ica_bovinos_2022.xlsx", "BOVINOS Y PREDIOS"),
    2023: (RAW / "ica" / "ica_bovinos_2023.xls", "BOVINOS Y PREDIOS"),
    2024: (RAW / "ica" / "ica_bovinos_2024.xlsx", "BOVINOS Y PREDIOS"),
    2025: (RAW / "ica" / "ica_bovinos_2025.xlsx", "BOVINOS Y PREDIOS"),
    2026: (RAW / "ica" / "ica_bovinos_2026.xlsx", "BOVINOS Y PREDIOS"),
}
CATEGORIES = ["f_lt1", "m_lt1", "f_1_2", "m_1_2", "f_2_3", "m_2_3", "f_gt3", "m_gt3"]
PATTERNS = [  # (regex on normalized header, standard name); first match wins
    (r"^DEPARTAMENTO", "dept"),
    (r"^CODIGO MUNICIPIO", "code"),
    (r"^MUNICIPIO", "mun"),
    (r"^TERNERAS", "f_lt1"),
    (r"^TERNEROS", "m_lt1"),
    (r"^HEMBRAS 1 2", "f_1_2"),
    (r"^MACHOS 1 2", "m_1_2"),
    (r"^HEMBRAS 2 3", "f_2_3"),
    (r"^MACHOS 2 3", "m_2_3"),
    (r"^HEMBRAS 3", "f_gt3"),
    (r"^MACHOS 3", "m_gt3"),
    (r"^TOTAL BOVINOS", "total"),
    (r"FINCAS 1 A 50", "farms_1_50"),
    (r"FINCAS 51 A 100", "farms_51_100"),
    (r"FINCAS 101 A 500", "farms_101_500"),
    (r"FINCAS 501", "farms_501_plus"),
    (r"^TOTAL FINCAS", "farms_total"),
    (r"^JUSTIFICACION", "note"),
]


# Department spellings used by ICA that differ from the DANE MGN names (verified by
# inspection of the 2017-2019 files).
DEPT_ALIASES = {
    "NORTE SANTANDER": "54", "VALLE": "76", "DISTRITO CAPITAL": "11", "BOGOTA D C": "11",
    "S ANDRES PROVID": "88", "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y": "88",
    "SAN ANDRES": "88", "SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "88",
    "SAN ANDRES Y PROVIDENCIA": "88",
}


def norm(s: object) -> str:
    """Upper-case ASCII name with punctuation collapsed to single spaces.

    The 2017-2019 ICA files append a two-letter department tag to homonymous
    municipalities (e.g. 'ARGELIA-An', 'MORALES-Bl'); the tag is removed.
    """
    s = re.sub(r"[-_ ]([A-Z][a-z]|SAYP)$", "", str(s).strip())
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).upper().strip()
    return re.sub(r"\s+", " ", s)


def read_year(year: int) -> pd.DataFrame:
    path, sheet = FILES[year]
    if year == 2019 and not path.exists():
        INTERIM.mkdir(parents=True, exist_ok=True)
        with (RAW / "ica" / "ica_bovinos_2019.xls").open("rb") as fh:
            of = msoffcrypto.OfficeFile(fh)
            of.load_key(password="VelvetSweatshop")  # Excel default write-protection key
            with path.open("wb") as out:
                of.decrypt(out)
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
    hdr = next(i for i in range(20) if any(norm(v) == "MUNICIPIO" for v in raw.iloc[i]))
    cols = {}
    for j, v in enumerate(raw.iloc[hdr]):
        n = norm(v)
        for pat, std in PATTERNS:
            if n and re.search(pat, n) and std not in cols.values():
                cols[j] = std
                break
    df = raw.iloc[hdr + 1:, list(cols)].copy()
    df.columns = list(cols.values())
    df["dept"] = df["dept"].ffill()
    df = df[df["mun"].notna() & ~df["mun"].astype(str).str.upper().str.contains("TOTAL")]
    df = df[~df["dept"].astype(str).str.upper().str.contains("TOTAL")]
    df.insert(0, "year", year)
    df["source_header_total"] = norm(raw.iloc[hdr][[k for k, v in cols.items() if v == "total"][0]])
    return df


def main() -> None:
    QC.mkdir(parents=True, exist_ok=True)
    mgn = gpd.read_file(f"zip://{RAW / 'dane' / 'MGN2024_MPIO_POLITICO.zip'}!MGN_ADM_MPIO_GRAFICO.shp",
                        ignore_geometry=True)
    mgn["code"] = mgn["mpio_cdpmp"].astype(str).str.zfill(5)
    dept_code = {norm(n): c for n, c in zip(mgn["dpto_cnmbr"], mgn["dpto_ccdgo"])}
    dept_code.update(DEPT_ALIASES)

    frames = [read_year(y) for y in FILES]
    df = pd.concat(frames, ignore_index=True)
    num_cols = CATEGORIES + ["total", "farms_1_50", "farms_51_100", "farms_101_500",
                             "farms_501_plus", "farms_total"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dept_n"], df["mun_n"] = df["dept"].map(norm), df["mun"].map(norm)
    df["code"] = df["code"].map(lambda v: None if pd.isna(v) else str(v).split(".")[0].zfill(5))
    df["dcode"] = df["dept_n"].map(dept_code)
    unknown_depts = sorted(df.loc[df["dcode"].isna(), "dept_n"].unique())
    if unknown_depts:
        raise ValueError(f"department names without code: {unknown_depts}")

    # Name -> code dictionary from years that report codes, plus MGN names.
    coded = df.dropna(subset=["code"])
    ref = pd.DataFrame({"dcode": coded["code"].str[:2], "mun_n": coded["mun_n"], "code": coded["code"]})
    ref = pd.concat([ref, pd.DataFrame({"dcode": mgn["dpto_ccdgo"].astype(str).str.zfill(2),
                                        "mun_n": mgn["mpio_cnmbr"].map(norm),
                                        "code": mgn["code"]})]).drop_duplicates()
    if XWALK.exists():
        xw = pd.read_csv(XWALK, dtype=str)
        xw["mun_n"] = xw["ica_name"].map(norm)
        ref = pd.concat([ref, xw[["dcode", "mun_n", "code"]]]).drop_duplicates()
    ambiguous = ref.groupby(["dcode", "mun_n"])["code"].nunique().loc[lambda s: s > 1]
    lookup = ref.drop_duplicates(["dcode", "mun_n"]).set_index(["dcode", "mun_n"])["code"]
    miss = df["code"].isna()
    df.loc[miss, "code"] = [lookup.get((d, m)) for d, m in zip(df.loc[miss, "dcode"], df.loc[miss, "mun_n"])]
    df["code_source"] = "file"
    df.loc[miss & df["code"].notna(), "code_source"] = "name_match"

    # Fuzzy match of the remaining names within the same department (audited).
    still = df["code"].isna()
    fuzzy_log = []
    for idx in df.index[still]:
        d, m = df.at[idx, "dcode"], df.at[idx, "mun_n"]
        cands = ref[ref["dcode"] == d].drop_duplicates("code")
        best = difflib.get_close_matches(m, cands["mun_n"].tolist(), n=2, cutoff=0.80)
        if best:
            code = cands.loc[cands["mun_n"] == best[0], "code"].iloc[0]
            score = difflib.SequenceMatcher(None, m, best[0]).ratio()
            second = (difflib.SequenceMatcher(None, m, best[1]).ratio() if len(best) > 1 else 0.0)
            if score - second >= 0.05:  # require a clear winner
                df.at[idx, "code"], df.at[idx, "code_source"] = code, "fuzzy_match"
            fuzzy_log.append((df.at[idx, "year"], d, m, best[0], code, round(score, 3),
                              round(second, 3), score - second >= 0.05))
    pd.DataFrame(fuzzy_log, columns=["year", "dcode", "ica_name", "ref_name", "code", "score",
                                     "second_score", "accepted"]).to_csv(
        QC / "ica_fuzzy_matches.csv", index=False)

    unmatched = df[df["code"].isna()][["year", "dept", "mun", "dept_n", "mun_n", "total"]]
    unmatched.to_csv(QC / "ica_unmatched_names.csv", index=False)
    pd.Series(ambiguous).to_csv(QC / "ica_ambiguous_names.csv")

    # QA tables
    df["cat_sum"] = df[CATEGORIES].sum(axis=1, min_count=1)
    df["flag_cat_sum_mismatch"] = (df["cat_sum"] - df["total"]).abs() > 0.5
    df["flag_non_integer"] = df[num_cols].apply(lambda s: (s.dropna() % 1 != 0).reindex(s.index, fill_value=False)).any(axis=1)
    df["flag_negative"] = (df[num_cols] < 0).any(axis=1)
    df["flag_code_not_in_mgn"] = df["code"].notna() & ~df["code"].isin(mgn["code"])
    df["flag_dept_mismatch"] = df["code"].notna() & (df["code"].str[:2] != df["dcode"])
    dup = df[df.duplicated(["year", "code"], keep=False) & df["code"].notna()]
    dup.to_csv(QC / "ica_duplicates_year_code.csv", index=False)

    summary = df.groupby("year").agg(
        rows=("mun", "size"), coded=("code", lambda s: s.notna().sum()),
        coded_by_name=("code_source", lambda s: (s == "name_match").sum()),
        coded_by_fuzzy=("code_source", lambda s: (s == "fuzzy_match").sum()),
        total_cattle=("total", "sum"), zero_total=("total", lambda s: (s == 0).sum()),
        missing_total=("total", lambda s: s.isna().sum()),
        farms=("farms_total", "sum"), cat_sum_mismatch=("flag_cat_sum_mismatch", "sum"),
        non_integer_rows=("flag_non_integer", "sum"), negative_rows=("flag_negative", "sum"),
        code_not_in_mgn=("flag_code_not_in_mgn", "sum"), dept_mismatch=("flag_dept_mismatch", "sum"),
        header_total_label=("source_header_total", "first"),
    )
    summary["duplicated_codes"] = dup.groupby("year").size().reindex(summary.index, fill_value=0)
    summary.to_csv(QC / "ica_summary_by_year.csv")
    print(summary.to_string())
    print("unmatched rows:", len(unmatched), "| ambiguous name keys:", len(ambiguous))

    keep = ["year", "code", "code_source", "dept", "mun"] + num_cols + [
        "flag_cat_sum_mismatch", "flag_non_integer", "flag_negative", "flag_code_not_in_mgn",
        "flag_dept_mismatch", "note"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
