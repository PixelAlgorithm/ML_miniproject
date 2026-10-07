"""Creates the shared train/val/test split (70/15/15, grouped by earthquake).
Run once after Member 1 pushes dataset/pairs_dataset.csv. Member 2 and Member 3 BOTH use the output.
Usage: python src/make_split.py [path_to_pairs_csv]
"""
import sys, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PAIRS = sys.argv[1] if len(sys.argv) > 1 else "dataset/pairs_dataset.csv"
GROUP, SEED, TEST, VAL = "EQID", 42, 0.15, 0.15

df = pd.read_csv(PAIRS)
g = df[GROUP].values
trval, te = next(GroupShuffleSplit(1, test_size=TEST, random_state=SEED).split(df, groups=g))
tr_rel, va_rel = next(GroupShuffleSplit(1, test_size=VAL/(1-TEST), random_state=SEED).split(df.iloc[trval], groups=g[trval]))
split = pd.Series("train", index=df.index, name="split")
split.iloc[trval[va_rel]] = "val"; split.iloc[te] = "test"
split.to_frame().to_csv("dataset/split_indices.csv", index_label="row")
print(split.value_counts())
