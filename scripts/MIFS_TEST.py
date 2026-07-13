"""
MIFS (Mutual Information Feature Selection) demo, following Section 3.4.4
of the El Nino proposal.

Pipeline:
  1. Build a synthetic ENSO-like dataset: T x N_phi x N_lambda x C tensor,
     flattened to a feature vector, with a continuous Nino 3.4-like target.
  2. Compute per-feature relevance I(X_ij; Y) via mutual_info_regression
     (k-NN / Kraskov estimator -> works directly on continuous variables).
  3. Greedy redundancy-penalized selection:
         S(X_ij) = I(X_ij; Y) - (1/|S|) * sum_{X_kl in S} I(X_ij; X_kl)
  4. Compare naive top-K-by-MI vs MIFS-selected features (redundancy check).
  5. Sweep K, train XGBoost, pick K that minimizes validation RMSE.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception as e:
    HAS_XGBOOST = False
    XGB_IMPORT_ERROR = e

rng = np.random.default_rng(42)


def make_model():
    """Use XGBoost when available; otherwise fall back to sklearn HGBR."""
    if HAS_XGBOOST:
        return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, verbosity=0)
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.1, max_iter=200, random_state=42)

# ---------------------------------------------------------------------
# 1. Synthetic dataset (small grid so the demo runs fast; the proposal's
#    real grid is 12 x 30 x 100 x 4 = 144,000 features per sample)
# ---------------------------------------------------------------------
T, N_PHI, N_LAMBDA, C = 4, 6, 10, 3          # time steps, lat, lon, variables
N_SAMPLES = 500

# Continuous Nino 3.4-like target: smooth autocorrelated series
y = np.cumsum(rng.normal(0, 1, N_SAMPLES))
y = (y - y.mean()) / y.std()

# "Hot region" of contiguous grid cells for variable 0 (SST) that truly
# drives the target, each cell = y + independent noise -> mutually
# redundant (this is exactly what the redundancy penalty should catch)
hot_region = [(i, j) for i in range(1, 3) for j in range(3, 7)]   # 2x4 block
# A second, weaker informative region for variable 2 (OHC-like), spatially
# separate from the SST hot region
warm_region = [(i, j) for i in range(4, 6) for j in range(0, 2)]  # 2x2 block

X = rng.normal(0, 1, size=(N_SAMPLES, T, N_PHI, N_LAMBDA, C))    # pure noise base
for t in range(T):
    for (i, j) in hot_region:
        X[:, t, i, j, 0] = y * 1.5 + rng.normal(0, 0.4, N_SAMPLES)
    for (i, j) in warm_region:
        X[:, t, i, j, 2] = y * 0.7 + rng.normal(0, 0.6, N_SAMPLES)

feature_names = [f"t{t}_lat{i}_lon{j}_var{c}"
                  for t in range(T) for i in range(N_PHI)
                  for j in range(N_LAMBDA) for c in range(C)]
X_flat = X.reshape(N_SAMPLES, -1)
df = pd.DataFrame(X_flat, columns=feature_names)
df["nino34"] = y
print(f"Flattened feature count: {X_flat.shape[1]}  (real proposal: 144,000)")
if not HAS_XGBOOST:
    print("WARNING: xgboost unavailable; falling back to HistGradientBoostingRegressor.")
    print(f"Reason: {XGB_IMPORT_ERROR}")

# Chronological split, mirroring 1980-2018 / 2019-2022 / 2023-2025 ratios
n_train = int(0.6 * N_SAMPLES)
n_val = int(0.2 * N_SAMPLES)
train, val, test = df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]
Xtr, ytr = train[feature_names].values, train["nino34"].values
Xval, yval = val[feature_names].values, val["nino34"].values
Xte, yte = test[feature_names].values, test["nino34"].values

# ---------------------------------------------------------------------
# 2. Relevance: I(X_ij; Y) for every feature, computed once
# ---------------------------------------------------------------------
relevance = mutual_info_regression(Xtr, ytr, random_state=42)
rel_series = pd.Series(relevance, index=feature_names).sort_values(ascending=False)

# ---------------------------------------------------------------------
# 3. Greedy, redundancy-penalized MIFS
#    (restrict candidate pool to top-150 by relevance first -- a standard
#    practical shortcut so redundancy MI isn't computed against all
#    144,000 candidates every iteration)
# ---------------------------------------------------------------------
POOL_SIZE, K_MAX = 150, 30
pool = rel_series.head(POOL_SIZE).index.tolist()
pool_idx = {name: i for i, name in enumerate(pool)}
Xpool = train[pool].values

selected, remaining = [], set(pool)
mi_cache = {}  # (candidate, selected) -> MI, avoid recomputation

def mi_between(a, b):
    key = (a, b) if a < b else (b, a)
    if key not in mi_cache:
        mi_cache[key] = mutual_info_regression(
            Xpool[:, [pool_idx[a]]], Xpool[:, pool_idx[b]], random_state=42
        )[0]
    return mi_cache[key]

while len(selected) < K_MAX:
    best_feat, best_score = None, -np.inf
    for cand in remaining:
        if not selected:
            score = rel_series[cand]
        else:
            redundancy = np.mean([mi_between(cand, s) for s in selected])
            score = rel_series[cand] - redundancy
        if score > best_score:
            best_feat, best_score = cand, score
    selected.append(best_feat)
    remaining.remove(best_feat)

print(f"\nTop 10 MIFS-selected features (redundancy-penalized):")
for f in selected[:10]:
    print(f"  {f}  relevance={rel_series[f]:.4f}")

print(f"\nTop 10 naive top-MI features (no redundancy penalty):")
for f in rel_series.head(10).index:
    print(f"  {f}  relevance={rel_series[f]:.4f}")

# ---------------------------------------------------------------------
# 4. K-sweep: train XGBoost on top-K MIFS features, pick K minimizing
#    validation RMSE (exactly as Section 3.4.4 "Selection Criterion")
# ---------------------------------------------------------------------
results = []
for K in range(2, K_MAX + 1, 2):
    feats = selected[:K]
    model = make_model()
    model.fit(train[feats], ytr)
    pred_val = model.predict(val[feats])
    rmse = mean_squared_error(yval, pred_val) ** 0.5
    results.append((K, rmse))

results_df = pd.DataFrame(results, columns=["K", "val_rmse"])
best_K = int(results_df.loc[results_df.val_rmse.idxmin(), "K"])
print(f"\nValidation RMSE by K:\n{results_df.to_string(index=False)}")
print(f"\nSelected K = {best_K} (minimizes validation RMSE)")

# ---------------------------------------------------------------------
# 5. Final test-set evaluation with best K
# ---------------------------------------------------------------------
final_feats = selected[:best_K]
final_model = make_model()
final_model.fit(train[final_feats], ytr)
pred_test = final_model.predict(test[final_feats])
test_rmse = mean_squared_error(yte, pred_test) ** 0.5
test_acc = np.corrcoef(yte, pred_test)[0, 1]
print(f"\nTest RMSE = {test_rmse:.4f}   Test ACC (corr) = {test_acc:.4f}")
print(f"\nRecovered hot region cells among final selected features:")
for f in final_feats:
    if "var0" in f and any(f"lat{i}_lon{j}" in f for i, j in hot_region):
        print(f"  {f}")