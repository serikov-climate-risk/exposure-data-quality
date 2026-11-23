"""A synthetic commercial property portfolio for Germany in an OED-like layout, and a corruption step that injects known defects.

Everything about the portfolio is simulated: locations are drawn from the GeoNames postcode centroids (CC BY 4.0), values from
lognormal distributions; the code lists are ASSUMED short lists, not the official OED code tables. The point is to test a validation
step against defects of known type and number, not to describe a real book.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONSTRUCTION = {"WOOD": 0.9, "MASONRY": 1.0, "CONCRETE": 1.2, "STEEL": 1.1}            # code: cost per m2 factor (assumed)
OCCUPANCY = {"OFFICE": 2200, "RETAIL": 2500, "WAREHOUSE": 1400, "LIGHT_INDUSTRY": 1900, "HOTEL": 3000}   # code: EUR per m2 base (assumed)
STATES = None


def load_postcodes(path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t", header=None, encoding="utf-8", dtype={1: str},
                    names=["cc", "postcode", "place", "state", "state_code", "c1", "c1c", "district", "dc", "lat", "lon", "acc"])
    d["postcode"] = d["postcode"].str.zfill(5)
    return d[["postcode", "place", "state", "lat", "lon"]]


def clean_portfolio(pc: pd.DataFrame, n: int = 8000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pcs = pc.drop_duplicates("postcode").reset_index(drop=True)
    # larger weight for postcodes with more places (a crude stand-in for settlement size)
    w = pc.groupby("postcode").size().reindex(pcs["postcode"]).values.astype(float)
    w = w / w.sum()
    pick = rng.choice(len(pcs), n, p=w)
    p = pcs.iloc[pick].reset_index(drop=True)
    area = np.exp(rng.normal(6.6, 0.9, n)).round(0).clip(80, 60000)
    cons = rng.choice(list(CONSTRUCTION), n, p=[0.05, 0.45, 0.35, 0.15])
    occ = rng.choice(list(OCCUPANCY), n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    base = np.array([OCCUPANCY[o] for o in occ]) * np.array([CONSTRUCTION[c] for c in cons])
    bld = (area * base * np.exp(rng.normal(0, 0.15, n))).round(-2)
    df = pd.DataFrame({
        "LocNumber": np.arange(1, n + 1),
        "AccNumber": ["A%03d" % k for k in rng.integers(1, 121, n)],
        "LocName": ["Standort %d" % k for k in range(1, n + 1)],
        "StreetAddress": [f"{rng.choice(['Haupt', 'Bahnhof', 'Industrie', 'Schul', 'Garten', 'Berg'])}strasse {rng.integers(1, 120)}" for _ in range(n)],
        "PostalCode": p["postcode"], "City": p["place"], "State": p["state"], "CountryCode": "DE",
        "Latitude": (p["lat"] + rng.normal(0, 0.01, n)).round(5), "Longitude": (p["lon"] + rng.normal(0, 0.015, n)).round(5),
        "ConstructionCode": cons, "OccupancyCode": occ, "YearBuilt": rng.integers(1900, 2024, n), "NumberOfStoreys": rng.integers(1, 9, n),
        "FloorArea": area, "BuildingTIV": bld, "ContentsTIV": (bld * rng.uniform(0.1, 0.5, n)).round(-2), "BITIV": (bld * rng.uniform(0.05, 0.3, n)).round(-2),
        "LocCurrency": "EUR"})
    df["TIV"] = df["BuildingTIV"] + df["ContentsTIV"] + df["BITIV"]
    return df


# defect type -> share of rows (each row gets at most one defect)
RATES = {"missing_coords": 0.02, "swapped_latlon": 0.01, "wrong_postcode_far": 0.02, "invalid_postcode": 0.01, "tiv_x1000": 0.006, "tiv_div1000": 0.004,
         "tiv_nonpositive": 0.005, "duplicate": 0.02, "year_invalid": 0.01, "storeys_invalid": 0.008, "code_unknown": 0.015, "currency_wrong": 0.005,
         "tiv_sum_mismatch": 0.01, "encoding": 0.01}


def corrupt(df: pd.DataFrame, rates: dict | None = None, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns the corrupted file and the truth log (one row per injected defect: row label, type). Duplicates are appended rows."""
    rates = rates or RATES
    rng = np.random.default_rng(seed)
    d = df.copy()
    n = len(d)
    order = rng.permutation(n)
    pos, log = 0, []
    for typ, r in rates.items():
        k = int(round(r * n))
        idx = order[pos:pos + k]
        pos += k
        lab = d.index[idx]
        if typ == "missing_coords":
            d.loc[lab, ["Latitude", "Longitude"]] = np.nan
        elif typ == "swapped_latlon":
            lat, lon = d.loc[lab, "Latitude"].copy(), d.loc[lab, "Longitude"].copy()
            d.loc[lab, "Latitude"], d.loc[lab, "Longitude"] = lon.values, lat.values
        elif typ == "wrong_postcode_far":
            # the coordinates belong to another region: take them from random other rows
            j = rng.integers(0, n, k)
            other = df["Latitude"].values[j], df["Longitude"].values[j]
            d.loc[lab, "Latitude"], d.loc[lab, "Longitude"] = other
        elif typ == "invalid_postcode":
            d.loc[lab, "PostalCode"] = [rng.choice(["00000", "1234", "ABCDE", "99999", "123456"]) for _ in lab]
        elif typ == "tiv_x1000":
            for c in ("BuildingTIV", "ContentsTIV", "BITIV", "TIV"):
                d.loc[lab, c] = d.loc[lab, c] * 1000
        elif typ == "tiv_div1000":
            for c in ("BuildingTIV", "ContentsTIV", "BITIV", "TIV"):
                d.loc[lab, c] = d.loc[lab, c] / 1000
        elif typ == "tiv_nonpositive":
            d.loc[lab, "BuildingTIV"] = rng.choice([0, -1], k)
            d.loc[lab, "TIV"] = d.loc[lab, "BuildingTIV"] + d.loc[lab, "ContentsTIV"] + d.loc[lab, "BITIV"]
        elif typ == "year_invalid":
            d.loc[lab, "YearBuilt"] = rng.choice([1066, 2099, 0, 9999], k)
        elif typ == "storeys_invalid":
            d.loc[lab, "NumberOfStoreys"] = rng.choice([0, -2, 150], k)
        elif typ == "code_unknown":
            d.loc[lab, "ConstructionCode"] = rng.choice(["UNK", "XX", "", "STONE?"], k)
        elif typ == "currency_wrong":
            d.loc[lab, "LocCurrency"] = rng.choice(["USD", "GBP", "EURO"], k)
        elif typ == "tiv_sum_mismatch":
            d.loc[lab, "TIV"] = d.loc[lab, "TIV"] * rng.uniform(1.2, 2.0, k)
        elif typ == "encoding":
            d.loc[lab, "City"] = d.loc[lab, "City"].astype(str).str.replace("ü", "Ã¼").str.replace("ö", "Ã¶").str.replace("ä", "Ã¤") + "Ã©"
        elif typ == "tiv_x3":
            for c in ("BuildingTIV", "ContentsTIV", "BITIV", "TIV"):
                d.loc[lab, c] = d.loc[lab, c] * 3
        elif typ == "coords_shift_15km":
            ang = rng.uniform(0, 2 * np.pi, k)
            d.loc[lab, "Latitude"] = d.loc[lab, "Latitude"] + 15 * np.cos(ang) / 111.0
            d.loc[lab, "Longitude"] = d.loc[lab, "Longitude"] + 15 * np.sin(ang) / (111.0 * np.cos(np.radians(d.loc[lab, "Latitude"])))
        elif typ == "code_swapped":
            d.loc[lab, "ConstructionCode"] = [rng.choice([c for c in CONSTRUCTION if c != x]) for x in d.loc[lab, "ConstructionCode"]]
        elif typ == "duplicate":
            pass
        if typ != "duplicate":
            for L in lab:
                log.append((L, typ))
    # duplicates: copies of existing rows, small value noise, new location number
    k = int(round(rates.get("duplicate", 0) * n))
    src = d.loc[d.index[order[pos:pos + k]]].copy()
    src["LocNumber"] = np.arange(n + 1, n + 1 + len(src))
    src["BuildingTIV"] = (src["BuildingTIV"] * rng.uniform(0.99, 1.01, len(src))).round(-2)
    src.index = np.arange(n, n + len(src))
    d = pd.concat([d, src])
    for L in src.index:
        log.append((L, "duplicate"))
    return d, pd.DataFrame(log, columns=["row", "defect"])


# defects that no rule of validate.py is written for: used to show what a rule-based pass cannot see
SUBTLE = {"tiv_x3": 0.02, "coords_shift_15km": 0.02, "code_swapped": 0.02}
