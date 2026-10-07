# Handoff: Member 2 -> Member 3

**Status:** Member 2 code is complete and runs on the real data (target = real delta_theta from Member 1).

## Run order (from repo root)
```
python src/prepare_pairs.py   # Member 1's dataset/station_pairs_delta_theta_5km.csv + cleaned csv -> dataset/pairs_dataset.csv
python src/make_split.py      # -> dataset/split_indices.csv  (shared split)
```
Member 2's notebook: `02_benchmark_ridge.ipynb`. **Do not make your own split** - read `dataset/split_indices.csv`
(column `split` = train/val/test, row-aligned with `pairs_dataset.csv`).

## Data
- Target `delta_theta` (deg, 0-90 axis), group column `EQID`, distance `site_distance_km`. 6,227 pairs (<5 km apart), 86 earthquakes.
- Target is heavy-tailed (median ~0.9 deg, mean ~2.2, max 77) -> report MAE as well as R2; a single split is noisy.
- The split is grouped by EQID and chosen (by target distribution only) so train/val/test have similar mean target.
- One-hot encode categoricals (`drop_first=True`); fit `StandardScaler` on TRAIN rows only.

## Output format
Write `results/metrics_member3.csv` with columns `model,split,R2,MAE,RMSE` (same as `results/metrics_member2.csv`, whose models are: Mean predictor, Distance-only benchmark, Ridge (all features), Ridge (selected features)). Use the same split, same scaling rule, and report test + grouped CV.
Please also report grouped 5-fold CV (`GroupKFold(5)` on `EQID`) like `results/cv_member2_summary.csv`.

## Paper reference (test set)
Benchmark R2 0.025 / MAE 20.2 · Ridge 0.245 / 16.9 · Neural network 0.354 / 13.2 (random 80/10/10 split; ours is grouped by EQID, so expect lower scores).
