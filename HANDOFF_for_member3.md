# Handoff: Member 2 -> Member 3

**Status:** Member 2 code is complete and runs on Member 1's updated cleaned CSV (it contains `Theta.D (deg)`).

## Run order (from repo root)
```
python src/prepare_pairs.py   # cleaned CSV -> dataset/pairs_dataset.csv (pairs <5 km, target delta_theta)
python src/make_split.py      # -> dataset/split_indices.csv  (shared split)
```
Member 2's notebook: `02_benchmark_ridge.ipynb`. **Do not make your own split** - read `dataset/split_indices.csv`
(column `split` = train/val/test, row-aligned with `pairs_dataset.csv`; grouped by EQID).

## Data
- Target `delta_theta` = folded |Theta.D_i - Theta.D_j| (deg, 0-90). Group column `EQID`, distance `site_distance_km`.
- 6,222 pairs (<5 km apart), 86 earthquakes; split 4187 / 1274 / 761 pairs (60 / 13 / 13 earthquakes).
- Target is heavy-tailed (median ~0.9 deg, mean ~2.1, max 77): report MAE and R2; a single split is noisy, so also report grouped 5-fold CV.
- Open question for the team: what `Theta.D` means (correlates 0.89 with source-to-site azimuth) - mention as a limitation.
- One-hot encode categoricals (`drop_first=True`); fit `StandardScaler` on TRAIN rows only. Do not use per-site PGA/PGV/PGD beyond `avg_pga_g`.

## Output format
`results/metrics_member3.csv` with columns `model,split,R2,MAE,RMSE` (same format as `results/metrics_member2.csv`; models there: Mean predictor, Distance-only benchmark, Ridge (all features), Ridge (selected features)). Also report `GroupKFold(5)` CV on EQID like `results/cv_member2_summary.csv`.

## Paper reference (test set)
Benchmark R2 0.025 / MAE 20.2 · Ridge 0.245 / 16.9 · Neural network 0.354 / 13.2 (random 80/10/10 split; ours is grouped by EQID).
