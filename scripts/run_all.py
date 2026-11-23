"""Generate the portfolio, corrupt it, validate, repair and measure detection and the effect on aggregate exposure."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edq import generate as G  # noqa: E402
from edq import validate as V  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
PC = G.load_postcodes(Path.home() / "z_data" / "z_exposure" / "DE.txt")
idx = V.postcode_index(PC)
clean = G.clean_portfolio(PC, n=8000, seed=0)
raw, truth = G.corrupt(clean, seed=1)
flags = V.validate(raw, idx)
flags.to_csv(RES / "flags.csv.gz")
MAP = {"missing_coords": ["missing_coords"], "swapped_latlon": ["swapped_latlon"], "wrong_postcode_far": ["postcode_far", "outside_country"],
       "invalid_postcode": ["invalid_postcode"], "tiv_x1000": ["tiv_per_m2_high"], "tiv_div1000": ["tiv_per_m2_low"], "tiv_nonpositive": ["tiv_nonpositive"],
       "duplicate": ["duplicate"], "year_invalid": ["year_invalid"], "storeys_invalid": ["storeys_invalid"], "code_unknown": ["code_unknown"],
       "currency_wrong": ["currency_wrong"], "tiv_sum_mismatch": ["tiv_sum_mismatch"], "encoding": ["encoding"]}
rows = []
for typ, rules in MAP.items():
    lab = truth.loc[truth.defect == typ, "row"].values
    hit = flags.loc[lab, rules].any(axis=1)
    flagged = flags[rules].any(axis=1)
    tp = int(hit.sum())
    rows.append({"defect": typ, "injected": int(len(lab)), "detected": tp, "recall": round(tp / len(lab), 3),
                 "rule_flags": int(flagged.sum()), "precision": round(tp / max(int(flagged.sum()), 1), 3)})
ev = pd.DataFrame(rows)
ev.to_csv(RES / "detection_by_defect.csv", index=False)
bad_rows = set(truth.row)
clean_rows = [i for i in raw.index if i not in bad_rows]
fp_rows = int(flags.loc[clean_rows].any(axis=1).sum())
out = {"locations_clean": int(len(clean)), "locations_file": int(len(raw)), "defects_injected": int(len(truth)),
       "rows_with_defect": int(truth.row.nunique()), "clean_rows_flagged": fp_rows, "clean_rows_flagged_share": round(fp_rows / len(clean_rows), 4),
       "detection": ev.to_dict("records"),
       "overall_recall": round(float(ev.detected.sum() / ev.injected.sum()), 3), "overall_precision": round(float(ev.detected.sum() / ev.rule_flags.sum()), 3)}

# effect on aggregate exposure
fixed = V.repair(raw, flags)
sev = pd.Series(V.RULES)
blocked = flags[[c for c in flags if sev[c] == "block"]].any(axis=1)
removed = blocked | flags["duplicate"]
final = fixed[~removed]


def cell_max_share(d):
    ok = d.dropna(subset=["Latitude", "Longitude"])
    ok = ok[ok.Latitude.between(*V.DE_LAT) & ok.Longitude.between(*V.DE_LON)]
    cell = (ok.Latitude // 0.5).astype(int).astype(str) + "_" + (ok.Longitude // 0.5).astype(int).astype(str)
    s = ok.groupby(cell).TIV.sum()
    return float(s.max() / s.sum()), float(s.max())


tot_clean = float(clean.TIV.sum())
c_share, c_max = cell_max_share(clean)
r_share, r_max = cell_max_share(raw)
f_share, f_max = cell_max_share(final)
out["exposure"] = {"total_tiv_clean_eur_bn": round(tot_clean / 1e9, 3), "total_tiv_raw_eur_bn": round(float(raw.TIV.sum()) / 1e9, 3),
                   "total_tiv_after_repair_and_removal_eur_bn": round(float(final.TIV.sum()) / 1e9, 3),
                   "raw_over_clean": round(float(raw.TIV.sum()) / tot_clean, 3), "after_over_clean": round(float(final.TIV.sum()) / tot_clean, 4),
                   "rows_removed": int(removed.sum()), "rows_removed_share": round(float(removed.mean()), 4),
                   "largest_half_degree_cell_share_tiv": {"clean": round(c_share, 4), "raw": round(r_share, 4), "after": round(f_share, 4)},
                   "largest_cell_tiv_eur_bn": {"clean": round(c_max / 1e9, 3), "raw": round(r_max / 1e9, 3), "after": round(f_max / 1e9, 3)}}
# remaining defects after the pass: injected defects that were not detected
undetected = []
for typ, rules in MAP.items():
    lab = truth.loc[truth.defect == typ, "row"].values
    miss = lab[~flags.loc[lab, rules].any(axis=1).values]
    undetected.append({"defect": typ, "missed": int(len(miss))})
out["missed"] = undetected
# defects the rules were not written for
raw2, truth2 = G.corrupt(clean, rates=G.SUBTLE, seed=2)
fl2 = V.validate(raw2, idx)
sub = []
for typ in G.SUBTLE:
    lab = truth2.loc[truth2.defect == typ, "row"].values
    hit = fl2.loc[lab].any(axis=1)
    sub.append({"defect": typ, "injected": int(len(lab)), "detected": int(hit.sum()), "recall": round(float(hit.mean()), 3)})
out["not_covered_by_rules"] = sub
print(sub)
(RES / "summary.json").write_text(json.dumps(out, indent=1))
print(ev.to_string())
print(json.dumps({k: out[k] for k in ("locations_file", "defects_injected", "clean_rows_flagged", "clean_rows_flagged_share", "overall_recall", "overall_precision", "exposure")}, indent=1))
