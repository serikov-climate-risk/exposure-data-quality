"""Rule-based validation of a location exposure file: one flag column per rule, a severity and, where unambiguous, a proposed repair.

The rules are written for a German commercial property file in an OED-like layout (see generate.py). Thresholds are parameters.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .generate import CONSTRUCTION, OCCUPANCY

DE_LAT, DE_LON = (47.0, 55.2), (5.5, 15.5)
RULES = {  # rule: severity (block = the location cannot be modelled as it is, fix = modellable after repair, warn = check)
    "missing_coords": "block", "outside_country": "block", "swapped_latlon": "fix", "postcode_far": "block", "invalid_postcode": "block",
    "tiv_nonpositive": "block", "tiv_per_m2_high": "fix", "tiv_per_m2_low": "fix", "tiv_peer_outlier": "warn", "duplicate": "warn", "year_invalid": "warn", "storeys_invalid": "warn",
    "code_unknown": "warn", "currency_wrong": "block", "tiv_sum_mismatch": "fix", "encoding": "fix"}
MOJIBAKE = re.compile("Ã[¼¶¤©Ÿ]")


def postcode_index(pc: pd.DataFrame) -> dict:
    return {k: g[["lat", "lon"]].values for k, g in pc.groupby("postcode")}


def _km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(a))


def validate(df: pd.DataFrame, pcidx: dict, far_km: float = 25.0, m2_lo: float = 150.0, m2_hi: float = 20000.0, peer_z: float = 4.0) -> pd.DataFrame:
    f = pd.DataFrame(False, index=df.index, columns=list(RULES))
    lat, lon = df["Latitude"], df["Longitude"]
    f["missing_coords"] = lat.isna() | lon.isna()
    swapped = (lat.between(*DE_LON) & lon.between(*DE_LAT)) & ~(lat.between(*DE_LAT) & lon.between(*DE_LON))
    f["swapped_latlon"] = swapped
    ok_c = ~f["missing_coords"] & ~swapped
    f["outside_country"] = ok_c & ~(lat.between(*DE_LAT) & lon.between(*DE_LON))
    pcs = df["PostalCode"].astype(str)
    valid_fmt = pcs.str.fullmatch(r"\d{5}")
    f["invalid_postcode"] = ~valid_fmt | ~pcs.isin(pcidx.keys())
    far = np.zeros(len(df), bool)
    la, lo = lat.values, lon.values
    for i, (p, a, b) in enumerate(zip(pcs.values, la, lo)):
        if p in pcidx and not np.isnan(a) and not np.isnan(b):
            if swapped.iloc[i]:
                a, b = b, a
            pts = pcidx[p]
            far[i] = _km(a, b, pts[:, 0], pts[:, 1]).min() > far_km
    f["postcode_far"] = far & ~f["outside_country"]
    tiv_parts = df["BuildingTIV"] + df["ContentsTIV"] + df["BITIV"]
    f["tiv_nonpositive"] = (df["BuildingTIV"] <= 0) | (df["TIV"] <= 0)
    per_m2 = df["BuildingTIV"] / df["FloorArea"]
    pos = ~f["tiv_nonpositive"]
    f["tiv_per_m2_high"] = pos & (per_m2 > m2_hi)
    f["tiv_per_m2_low"] = pos & (per_m2 < m2_lo)
    f["tiv_sum_mismatch"] = pos & ((df["TIV"] - tiv_parts).abs() > 0.02 * tiv_parts.abs().clip(lower=1)) & ~(f["tiv_per_m2_high"] | f["tiv_per_m2_low"])
    # robust z-score of log value per m2 against locations with the same occupancy and construction code
    lv = np.log(per_m2.where(pos & (per_m2 > 0)))
    g = df["OccupancyCode"].astype(str) + "|" + df["ConstructionCode"].astype(str)
    med = lv.groupby(g).transform("median")
    mad = (lv - med).abs().groupby(g).transform("median") * 1.4826
    z = (lv - med) / mad.clip(lower=0.02)
    f["tiv_peer_outlier"] = (z.abs() > peer_z) & ~(f["tiv_per_m2_high"] | f["tiv_per_m2_low"])
    key = df["StreetAddress"].astype(str) + "|" + pcs
    dup = pd.Series(False, index=df.index)
    for _, g in df.assign(_k=key).groupby("_k"):
        if len(g) > 1:
            g = g.sort_values("LocNumber")
            base = g["BuildingTIV"].iloc[0]
            later = g.iloc[1:]
            near = (later["BuildingTIV"] - base).abs() <= 0.05 * abs(base)
            dup.loc[later.index[near.values]] = True
    f["duplicate"] = dup
    f["year_invalid"] = ~df["YearBuilt"].between(1700, 2026)
    f["storeys_invalid"] = ~df["NumberOfStoreys"].between(1, 60)
    f["code_unknown"] = ~df["ConstructionCode"].isin(CONSTRUCTION) | ~df["OccupancyCode"].isin(OCCUPANCY)
    f["currency_wrong"] = df["LocCurrency"] != "EUR"
    f["encoding"] = df["City"].astype(str).str.contains(MOJIBAKE)
    return f


def repair(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Apply only unambiguous repairs: swap latitude and longitude, undo a factor-1000 unit error, fix mojibake, recompute TIV from its parts."""
    d = df.copy()
    s = flags["swapped_latlon"]
    d.loc[s, ["Latitude", "Longitude"]] = d.loc[s, ["Longitude", "Latitude"]].values
    hi, lo = flags["tiv_per_m2_high"], flags["tiv_per_m2_low"]
    for c in ("BuildingTIV", "ContentsTIV", "BITIV", "TIV"):
        d.loc[hi, c] = d.loc[hi, c] / 1000
        d.loc[lo, c] = d.loc[lo, c] * 1000
    e = flags["encoding"]
    d.loc[e, "City"] = [str(x).replace("Ã¼", "ü").replace("Ã¶", "ö").replace("Ã¤", "ä").replace("Ã©", "") for x in d.loc[e, "City"]]
    m = flags["tiv_sum_mismatch"]
    d.loc[m, "TIV"] = d.loc[m, ["BuildingTIV", "ContentsTIV", "BITIV"]].sum(axis=1)
    return d
