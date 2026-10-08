"""Builds the model-ready pair dataset DIRECTLY from Member 1's cleaned CSV (which now contains Theta.D (deg)).
One row per unordered pair of DIFFERENT stations of the SAME earthquake (EQID) that are closer than MAX_DIST_KM.
Target: delta_theta = folded |Theta.D_i - Theta.D_j|  (deg, 0..90)  - angles are axial, so theta and theta+180 are the same axis.
Input : dataset/cleaned_dataset_NGA_West2_d050.csv
Output: dataset/pairs_dataset.csv     Usage: python src/prepare_pairs.py   then   python src/make_split.py
PGA is kept only as the pair-average (avg_pga_g); PGV/PGD are not used.
"""
import itertools, numpy as np, pandas as pd

CLEAN, OUT = "dataset/cleaned_dataset_NGA_West2_d050.csv", "dataset/pairs_dataset.csv"
ANGLE, MAX_DIST_KM = "Theta.D (deg)", 5.0

def haversine_km(la1, lo1, la2, lo2):
    p = np.pi / 180
    a = np.sin((la2-la1)*p/2)**2 + np.cos(la1*p)*np.cos(la2*p)*np.sin((lo2-lo1)*p/2)**2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

d = pd.read_csv(CLEAN)
assert ANGLE in d.columns, f"'{ANGLE}' missing - use the updated cleaned CSV from Member 1"
rows = []
for eq, g in d.groupby("EQID"):
    g = g.drop_duplicates(subset=["Station Name", "Station Latitude", "Station Longitude"]).reset_index(drop=True)
    for i, j in itertools.combinations(range(len(g)), 2):          # i<j: each pair once, never a self-pair
        a, b = g.loc[i], g.loc[j]
        dist = haversine_km(a["Station Latitude"], a["Station Longitude"], b["Station Latitude"], b["Station Longitude"])
        if dist >= MAX_DIST_KM: continue
        diff = abs(a[ANGLE] - b[ANGLE]) % 180
        rows.append(dict(EQID=int(eq), site_distance_km=dist,
                         magnitude=a["Earthquake Magnitude"], depth_km=a["Hypocenter Depth (km)"],
                         strike=a["Strike (deg)"], dip=a["Dip (deg)"], rake=a["Rake Angle (deg)"],
                         mechanism=a["Mechanism Based on Rake Angle"],
                         avg_pga_g=(a["PGA (g)"] + b["PGA (g)"]) / 2,
                         avg_vs30=(a["Vs30 (m/s) selected for analysis"] + b["Vs30 (m/s) selected for analysis"]) / 2,
                         vs30_1=a["Vs30 (m/s) selected for analysis"], vs30_2=b["Vs30 (m/s) selected for analysis"],
                         clstd_1=a["ClstD (km)"], clstd_2=b["ClstD (km)"],
                         nehrp_1=a["Preferred NEHRP Based on Vs30"], nehrp_2=b["Preferred NEHRP Based on Vs30"],
                         delta_theta=min(diff, 180 - diff)))
p = pd.DataFrame(rows)
assert p.notna().all().all(), "unexpected missing values"
p.to_csv(OUT, index=False)
print("pairs:", p.shape, "| earthquakes:", p.EQID.nunique(), "| target: delta_theta from Theta.D (deg)")
print(p.delta_theta.describe().round(3).to_string())
