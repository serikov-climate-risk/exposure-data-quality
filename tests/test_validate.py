import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edq import generate as G  # noqa: E402
from edq import validate as V  # noqa: E402

PC = pd.DataFrame({"postcode": ["20095", "80331", "50667"], "place": ["Hamburg", "Muenchen", "Koeln"], "state": ["HH", "BY", "NW"],
                   "lat": [53.55, 48.14, 50.94], "lon": [10.0, 11.57, 6.96]})
IDX = V.postcode_index(PC)


def _row(**kw):
    r = {"LocNumber": 1, "AccNumber": "A1", "LocName": "x", "StreetAddress": "Hauptstrasse 1", "PostalCode": "20095", "City": "Hamburg", "State": "HH",
         "CountryCode": "DE", "Latitude": 53.55, "Longitude": 10.0, "ConstructionCode": "MASONRY", "OccupancyCode": "OFFICE", "YearBuilt": 1990,
         "NumberOfStoreys": 3, "FloorArea": 1000.0, "BuildingTIV": 2_200_000.0, "ContentsTIV": 300_000.0, "BITIV": 100_000.0, "LocCurrency": "EUR"}
    r.update(kw)
    r["TIV"] = kw.get("TIV", r["BuildingTIV"] + r["ContentsTIV"] + r["BITIV"])
    return r


def _flags(*rows):
    df = pd.DataFrame(list(rows))
    df["LocNumber"] = np.arange(1, len(df) + 1)
    df.index = np.arange(len(df))
    return V.validate(df, IDX), df


def test_clean_row_has_no_flags():
    f, _ = _flags(_row())
    assert not f.iloc[0].any()


def test_each_rule_fires_on_its_defect():
    cases = {"missing_coords": {"Latitude": np.nan}, "swapped_latlon": {"Latitude": 10.0, "Longitude": 53.55},
             "invalid_postcode": {"PostalCode": "1234"}, "tiv_nonpositive": {"BuildingTIV": 0.0},
             "year_invalid": {"YearBuilt": 2099}, "storeys_invalid": {"NumberOfStoreys": 0}, "code_unknown": {"ConstructionCode": "UNK"},
             "currency_wrong": {"LocCurrency": "USD"}, "tiv_sum_mismatch": {"TIV": 5_000_000.0}, "encoding": {"City": "KÃ¶ln"},
             "tiv_per_m2_high": {"BuildingTIV": 2_200_000_000.0, "ContentsTIV": 3e8, "BITIV": 1e8, "TIV": 2_600_000_000.0},
             "tiv_per_m2_low": {"BuildingTIV": 2_200.0, "ContentsTIV": 300.0, "BITIV": 100.0, "TIV": 2_600.0},
             "postcode_far": {"Latitude": 48.14, "Longitude": 11.57}}
    for rule, kw in cases.items():
        f, _ = _flags(_row(**kw))
        assert f[rule].iloc[0], rule


def test_swapped_coordinates_are_repaired_and_then_match_the_postcode():
    f, df = _flags(_row(Latitude=10.0, Longitude=53.55))
    fixed = V.repair(df, f)
    assert fixed.loc[0, "Latitude"] == 53.55 and not V.validate(fixed, IDX).iloc[0].any()
    assert not f["postcode_far"].iloc[0]


def test_unit_error_is_repaired():
    f, df = _flags(_row(BuildingTIV=2_200_000_000.0, ContentsTIV=300_000_000.0, BITIV=100_000_000.0, TIV=2_600_000_000.0))
    fixed = V.repair(df, f)
    assert np.isclose(fixed.loc[0, "BuildingTIV"], 2_200_000.0) and np.isclose(fixed.loc[0, "TIV"], 2_600_000.0)


def test_duplicate_flags_only_the_later_location():
    f, _ = _flags(_row(), _row(BuildingTIV=2_210_000.0))
    assert not f["duplicate"].iloc[0] and f["duplicate"].iloc[1]


def test_corruption_log_matches_injected_counts():
    rng = np.random.default_rng(0)
    clean = G.clean_portfolio(pd.concat([PC] * 3, ignore_index=True), n=500, seed=0)
    raw, log = G.corrupt(clean, rates={"missing_coords": 0.02, "duplicate": 0.02}, seed=1)
    assert (log.defect == "missing_coords").sum() == 10 and (log.defect == "duplicate").sum() == 10 and len(raw) == 510
