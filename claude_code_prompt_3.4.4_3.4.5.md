# Task: Implement MIFS Feature Selection (Sec 3.4.4) and XGBoost Baseline (Sec 3.4.5)

## Project context
This is the "AI-Enhanced Prediction of El Niño and its Regional Climatic Impacts"
minor project. The full pipeline predicts the Niño 3.4 index at a 6-month lead
time using a CNN-TCN as the primary model and XGBoost as a baseline. This task
covers only the XGBoost branch: feature selection (MIFS) and the baseline model
itself, exactly as specified in proposal Sections 3.4.4 and 3.4.5.

## Input data
A CSV in long format, one row per (time, latitude, longitude), with columns:
```
time, latitude, longitude,
avg_iews, avg_inss, ttr, sst, t2m, msl,                      # raw
avg_iews_anom, avg_inss_anom, ttr_anom, sst_anom, t2m_anom, msl_anom,   # anomaly
avg_iews_anom_z, avg_inss_anom_z, ttr_anom_z, sst_anom_z, t2m_anom_z, msl_anom_z,  # z-scored anomaly
lsm, land_ocean, land_ocean_flag,                            # static geography, ignore
so20chgt, sohtc300, sossheig,                                # ocean raw
so20chgt_anom, sohtc300_anom, sossheig_anom,                 # ocean anomaly
so20chgt_anom_z, sohtc300_anom_z, sossheig_anom_z            # ocean z-scored anomaly
```
A separate Niño 3.4 index CSV/series exists, one value per month, both in raw and
z-scored normalized form.

**Important discrepancy to flag before writing code**: proposal Section 3.4.3
specifies exactly C=4 channels (SST anomaly, SLP anomaly, wind stress anomaly,
OHC anomaly), but this CSV has 9 candidate `_anom_z` variables. Print a summary of
available variables and ask me to confirm the final channel set before proceeding,
rather than assuming which 4 (or whether all 9) should be used.

## Step 1 — Reshape into the tensor input (Sec 3.4.3 foundation)
1. Use only the `_anom_z` (z-scored anomaly) columns as candidate features. Do
   not use raw or anomaly-only columns for modeling — z-scored anomaly is the
   correct input for both MI estimation and XGBoost training.
2. Drop `lsm`, `land_ocean`, `land_ocean_flag` — static geography metadata with
   no time variation, not part of the feature set.
3. Pivot from long format to wide format: one row per month, one column per
   (variable, latitude, longitude) combination.
4. Drop columns that are entirely NaN across all time (e.g., `sst_anom_z` at
   land grid cells, ocean-only variables at land cells). Do not impute these —
   they have no physical value to recover. If a column has only sporadic NaNs,
   flag it and ask before choosing an imputation approach.
5. Build 12-month rolling windows (T=12, per Sec 3.4.3): for each window ending
   at month `t`, tag every column with `_lag0` (month `t`, most recent) through
   `_lag11` (month `t-11`, oldest), flattening the window into one row of
   `N_lat x N_lon x C x 12` columns.
6. Align the target: for a window ending at month `t`, the target is the Niño
   3.4 index (z-scored) at month `t + 6` (6-month lead time, per Sec 3.4.3/3.4.5).
   Confirm this lead-time convention against the team's exact definition before
   finalizing — verify whether "window end" or "window start" is the reference
   point for the 6-month offset.
7. Split chronologically, never randomly: train = 1980-2018, validation =
   2019-2022, test = 2023-2025. Assert no date overlap between splits given the
   rolling-window overlap (a window ending in Dec 2018 and one starting in
   Dec 2018 must not straddle the train/val boundary).

## Step 2 — MIFS feature selection (Sec 3.4.4), fit on TRAIN ONLY
Implement Mutual Information Feature Selection exactly as defined:

1. **Relevance** — for every candidate feature column, compute
   `I(X_ij; Y)` using a continuous-continuous MI estimator (e.g.
   `sklearn.feature_selection.mutual_info_regression`, Kraskov k-NN estimator).
   Compute this once, using training data only.
2. **Redundancy-penalized score** — for each candidate not yet selected:
   `S(X_ij) = I(X_ij; Y) - (1/|S|) * sum_{X_kl in S} I(X_ij; X_kl)`
   where `S` is the current selected set. On the first iteration (S empty),
   score = relevance only.
3. **Greedy loop** — repeatedly select the candidate with the highest `S(X_ij)`,
   add it to `S`, and recompute scores for remaining candidates (redundancy
   term grows as `S` grows). For efficiency at full scale (144,000 candidates),
   first restrict the candidate pool to the top ~150-300 by raw relevance
   before running the greedy redundancy loop — cache pairwise MI computations
   to avoid recomputation. Flag if this shortcut appears to exclude a variable
   type entirely (e.g., a weaker predictor like OHC never entering the pool)
   since that is a known limitation worth reporting, not silently ignoring.
4. Do not use validation or test data anywhere in relevance or redundancy
   computation — MIFS is fit exclusively on the training split, then the
   resulting feature name list is simply used to slice matching columns out of
   validation and test (no recomputation on those splits).
5. **K selection** — sweep K (e.g., steps of 2 up to some max like 50), train
   an XGBoost model on the top-K MIFS features at each K, and evaluate RMSE on
   the validation set (2019-2022). Select the K that minimizes validation RMSE.
6. Save the final selected feature name list and the full relevance/redundancy
   ranking to disk (e.g., CSV or JSON) so selection is reproducible and
   auditable without rerunning MIFS.
7. Report a physical sanity check: print which variables and approximate
   regions dominate the selected set, and compare against the proposal's
   expectation (central/eastern Pacific SST, Walker Circulation wind stress,
   western Pacific thermocline OHC should dominate).

## Step 3 — XGBoost baseline model (Sec 3.4.5)
1. Train XGBoost regression on the training period (1980-2018) using the final
   MIFS-selected K features, predicting the z-scored Niño 3.4 index at 6-month
   lead time.
2. Tune hyperparameters (`n_estimators`, `max_depth`, `learning_rate`) via grid
   search, evaluated on the validation set (2019-2022).
3. Evaluate final performance on the held-out test set (2023-2025) only once,
   using the best hyperparameters and best K from validation. Report both:
   - Root Mean Square Error (RMSE)
   - Anomaly Correlation Coefficient (ACC) — Pearson correlation between
     predicted and actual Niño 3.4 index, consistent with the metric used for
     the CNN-TCN model.
4. De-normalize predictions and targets back to physical Niño 3.4 units (using
   the training-set-derived mean/std saved during normalization) before final
   reporting, so RMSE/ACC are interpretable in real index units, not z-scores.
5. Save: selected feature list, best K, best hyperparameters, validation RMSE
   curve across K, and final test-set metrics — all to disk for inclusion in
   the project report.

## Deliverables
- A data pipeline module (reshape long → wide → rolling window → split)
- A MIFS module (relevance, redundancy, greedy selection, K-sweep)
- A baseline model module (XGBoost training, grid search, final evaluation)
- Clear docstrings/comments referencing the proposal's Section 3.4.4 equations
  (3-8 relevance, 3-9 redundancy-penalized score) so the code and report stay
  aligned
- A short printed/logged summary at the end: channels used, K chosen, best
  hyperparameters, test RMSE, test ACC

## Constraints
- No data leakage: normalization statistics, climatology, and MIFS selection
  must all be fit on the training split only, then applied (not refit) to
  validation and test.
- Chronological splits only — never random shuffling across time.
- Prefer `xarray` over `pandas` for any gridded intermediate representations,
  per existing project convention, but `pandas`/NumPy are fine once flattened
  to the tabular MIFS/XGBoost input.
- Keep code concise, well-commented, and modular rather than one long script.

## Paths for datasets
- Input CSV: `data/combined_era5_oras5.csv`
- Niño 3.4 index CSV: `d/Users/raman/Elnino/data/ERA5/nino34_index_monthly.csv`