"""Builds the site-pair dataset from Member 1's cleaned CSV.
One row per unordered pair of DIFFERENT stations that recorded the SAME earthquake (EQID).
Adds: site_distance_km (haversine), pair features, and delta_theta IF a per-station angle column exists.

Usage: python src/make_pairs.py
Output: dataset/pairs_dataset.csv
"""
import itertools, numpy as np, pandas as pd

IN, OUT = "dataset/cleaned_dataset_NGA_West2_d050.csv", "dataset/pairs_dataset.csv"
ANGLE_COL = "theta_deg"   # per-station direction of max shaking (degrees). NOT in the flatfile -> must come from elsewhere.

def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2-lat1)*p/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

d = pd.read_csv(IN)
rows = []
for eq, g in d.groupby("EQID"):
    g = g.drop_duplicates(subset=["Station Name", "Station Latitude", "Station Longitude"]).reset_index(drop=True)
    for i, j in itertools.combinations(range(len(g)), 2):      # i<j -> each pair once, never self-pair
        a, b = g.loc[i], g.loc[j]
        r = dict(EQID=eq,
                 site_distance_km=haversine_km(a["Station Latitude"], a["Station Longitude"], b["Station Latitude"], b["Station Longitude"]),
                 magnitude=a["Earthquake Magnitude"], depth_km=a["Hypocenter Depth (km)"],
                 strike=a["Strike (deg)"], dip=a["Dip (deg)"], rake=a["Rake Angle (deg)"],
                 mechanism=a["Mechanism Based on Rake Angle"],
                 vs30_1=a["Vs30 (m/s) selected for analysis"], vs30_2=b["Vs30 (m/s) selected for analysis"],
                 clstd_1=a["ClstD (km)"], clstd_2=b["ClstD (km)"],
                 nehrp_1=a["Preferred NEHRP Based on Vs30"], nehrp_2=b["Preferred NEHRP Based on Vs30"])
        if ANGLE_COL in g.columns:
            diff = abs(a[ANGLE_COL] - b[ANGLE_COL]) % 180      # axial data: theta and theta+180 are the same axis
            r["delta_theta"] = min(diff, 180 - diff)           # range 0..90
        rows.append(r)

p = pd.DataFrame(rows)
if "delta_theta" not in p.columns:
    p["delta_theta"] = np.nan
    print("WARNING: no per-station angle column found -> delta_theta is EMPTY. Models cannot be trained yet.")
p.to_csv(OUT, index=False)
print("pairs:", p.shape, "| earthquakes:", p.EQID.nunique(), "| distance km: %.1f-%.1f" % (p.site_distance_km.min(), p.site_distance_km.max()))
