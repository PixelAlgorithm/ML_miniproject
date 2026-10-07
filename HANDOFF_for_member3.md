# Handoff: Member 2 -> Member 3

**Status:** Member 2 code (`02_benchmark_ridge.ipynb`) is complete. Real results wait only on Member 1's `dataset/pairs_dataset.csv`.

## What you can use
- **Same data:** `dataset/pairs_dataset.csv` (columns: `EQID`, `delta_theta` target, `site_distance_km`, pair features).
- **Same split:** `python src/make_split.py` -> `dataset/split_indices.csv` (column `split` = train/val/test, row-aligned with the pairs file; grouped by `EQID`, seed 42). Do NOT make your own split.
- **Same preprocessing:** one-hot encode categoricals (`drop_first=True`); fit `StandardScaler` on the TRAIN rows only.
- **Same metrics:** R2, MAE, RMSE. Write yours to `results/metrics_member3.csv` with columns `model,split,R2,MAE,RMSE`, same format as `results/metrics_member2.csv` (models: "Mean predictor", "Distance-only benchmark", "Ridge Regression"), so the final comparison is one concat.

## Start today (before real data)
`python src/make_fake_pairs.py` creates `dataset/pairs_dataset_FAKE.csv` (random target). Point your code at it to debug. Switch the path to the real file when it lands. Never report FAKE results.
