"""Creates the SHARED train/val/test split (about 70/15/15), grouped by earthquake (EQID).
Used by Member 2 AND Member 3 -> run once, commit dataset/split_indices.csv, never re-make it separately.

Why a seed search: with few earthquakes and a heavy-tailed target, one random split can put a single
unusual earthquake in the test set. Among seeds 0..299 we pick the split whose train/val/test have the most similar
MEAN TARGET (uses only the target distribution, never model performance).
Usage: python src/make_split.py [pairs_csv]
"""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PAIRS = sys.argv[1] if len(sys.argv) > 1 else "dataset/pairs_dataset.csv"
GROUP, TEST, VAL = "EQID", 0.15, 0.15
df = pd.read_csv(PAIRS)
TARGET = "delta_theta" if "delta_theta" in df.columns else "delta_ln_pga"
y, g = df[TARGET].values, df[GROUP].values

def make(seed):
    trval, te = next(GroupShuffleSplit(1, test_size=TEST, random_state=seed).split(df, groups=g))
    tr_rel, va_rel = next(GroupShuffleSplit(1, test_size=VAL/(1-TEST), random_state=seed).split(df.iloc[trval], groups=g[trval]))
    return trval[tr_rel], trval[va_rel], te

best = None
for seed in range(300):
    tr, va, te = make(seed)
    if min(len(va), len(te)) < 0.12 * len(df): continue          # keep val/test reasonably large
    m = [y[i].mean() for i in (tr, va, te)]
    score = max(abs(x - y.mean()) for x in m) / y.mean()          # relative spread of split means
    if best is None or score < best[0]: best = (score, seed, tr, va, te)
score, seed, tr, va, te = best
split = pd.Series("train", index=df.index, name="split"); split.iloc[va] = "val"; split.iloc[te] = "test"
split.to_frame().to_csv("dataset/split_indices.csv", index_label="row")
print(f"chosen seed {seed} (relative mean spread {score:.3f}); target={TARGET}")
print(pd.DataFrame({"pairs": split.value_counts(), "mean_target": df.groupby(split.values)[TARGET].mean().round(2),
                    "earthquakes": df.groupby(split.values)[GROUP].nunique()}))
