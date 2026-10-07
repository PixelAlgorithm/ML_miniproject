"""Turns Member 1's REAL pair file (with true delta_theta) into the model-ready dataset.
Input : dataset/station_pairs_delta_theta_5km.csv   (Member 1: pairs <5 km apart with Theta_1/2 and Delta_Theta)
        dataset/cleaned_dataset_NGA_West2_d050.csv  (extra per-site source/site features, joined on RSN)
Output: dataset/pairs_dataset.csv  (target = delta_theta, group = EQID, distance = site_distance_km)
Usage : python src/prepare_pairs.py     then   python src/make_split.py
NOTE  : PGA/PGV/PGD per site are not added; only Member 1's Avg_PGA_g is kept as a feature (it does not build the target).
"""
import pandas as pd

PAIRS = "dataset/station_pairs_delta_theta_5km.csv"
CLEAN = "dataset/cleaned_dataset_NGA_West2_d050.csv"
OUT   = "dataset/pairs_dataset.csv"

p = pd.read_csv(PAIRS)
c = pd.read_csv(CLEAN).set_index("Record Sequence Number")

src = ["Hypocenter Depth (km)", "Strike (deg)", "Dip (deg)", "Rake Angle (deg)", "Mechanism Based on Rake Angle"]
site = ["Vs30 (m/s) selected for analysis", "ClstD (km)", "Preferred NEHRP Based on Vs30"]

out = pd.DataFrame({"EQID": p["EQID"].astype(int),
                    "site_distance_km": p["Inter_Station_Distance_km"],
                    "magnitude": p["Earthquake_Magnitude"],
                    "avg_pga_g": p["Avg_PGA_g"], "avg_vs30": p["Avg_Vs30_mps"]})
for col, name in zip(src, ["depth_km", "strike", "dip", "rake", "mechanism"]):
    out[name] = p["Station_1_RSN"].map(c[col]).values            # earthquake-level -> same for both sites
for col, name in zip(site, ["vs30", "clstd", "nehrp"]):
    out[name + "_1"] = p["Station_1_RSN"].map(c[col]).values
    out[name + "_2"] = p["Station_2_RSN"].map(c[col]).values
out["delta_theta"] = p["Delta_Theta_deg"]

assert out.notna().all().all(), "unexpected missing values after merge"
out.to_csv(OUT, index=False)
print("pairs:", out.shape, "| earthquakes:", out.EQID.nunique(), "| target: delta_theta (REAL angle difference, deg)")
print(out.delta_theta.describe().round(3).to_string())
