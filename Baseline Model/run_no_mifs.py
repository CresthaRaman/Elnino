"""
XGBoost Baseline — No MIFS, all 260K features directly.
Saves to pipeline_output_no_mifs/ for comparison.
"""

import numpy as np
import json
from pathlib import Path
from itertools import product
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pipeline import run_pipeline
from xgboost_baseline import anomaly_correlation

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "pipeline_output_no_mifs"

PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5],
    "learning_rate": [0.1],
}

# ─── Pipeline ────────────────────────────────────────────────────────────────

print("=" * 70)
print("STEP 1: Data Pipeline")
print("=" * 70)
splits, _ = run_pipeline(save=False)

X_train, y_train, _ = splits["train"]
X_val, y_val, _ = splits["val"]
X_test, y_test, times_test = splits["test"]

# Convert to arrays, replace NaN
X_tr = np.nan_to_num(X_train.values, nan=0.0)
X_v = np.nan_to_num(X_val.values, nan=0.0)
X_te = np.nan_to_num(X_test.values, nan=0.0)
y_tr = y_train.values
y_v = y_val.values
y_te = y_test.values

print(f"\nTrain: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")
print(f"Using ALL {X_tr.shape[1]:,} features (no MIFS)")

# ─── Grid search ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 2: Grid Search (no feature selection)")
print("=" * 70)

keys = list(PARAM_GRID.keys())
combos = list(product(*PARAM_GRID.values()))
print(f"{len(combos)} combinations to evaluate")

grid_results = []
best_rmse = np.inf
best_params = None

for i, values in enumerate(combos):
    params = dict(zip(keys, values))
    model = XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=42, verbosity=0,
        tree_method="hist",
        colsample_bytree=0.01,  # sample 1% of 260K features per tree
        subsample=0.8,
    )
    model.fit(X_tr, y_tr)
    yp = model.predict(X_v)
    rmse = np.sqrt(mean_squared_error(y_v, yp))
    acc = anomaly_correlation(y_v, yp)
    grid_results.append({**params, "val_rmse": rmse, "val_acc": acc})
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params
    if (i + 1) % 16 == 0:
        print(f"  [{i+1}/{len(combos)}] best: RMSE={best_rmse:.4f} {best_params}")

print(f"\nBest params: {best_params}")
print(f"Best val RMSE: {best_rmse:.4f}")
grid_results.sort(key=lambda x: x["val_rmse"])

# ─── Final test evaluation ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 3: Final Test Evaluation")
print("=" * 70)

final_model = XGBRegressor(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    random_state=42, verbosity=0,
    tree_method="hist",
    colsample_bytree=0.01,
    subsample=0.8,
)
final_model.fit(X_tr, y_tr)
y_pred = final_model.predict(X_te)

test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
test_acc = anomaly_correlation(y_te, y_pred)

print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test ACC:  {test_acc:.4f}")

# ─── Save ────────────────────────────────────────────────────────────────────

import pandas as pd

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(grid_results).to_csv(OUTPUT_DIR / "xgb_grid_search.csv", index=False)

pd.DataFrame({
    "window_end": times_test,
    "actual_nino34": y_te,
    "predicted_nino34": y_pred,
}).to_csv(OUTPUT_DIR / "xgb_test_predictions.csv", index=False)

summary = {
    "model": "XGBRegressor (no MIFS, all features)",
    "n_features": int(X_tr.shape[1]),
    "best_params": best_params,
    "test_rmse": round(test_rmse, 4),
    "test_acc": round(test_acc, 4),
    "train_samples": len(y_tr),
    "val_samples": len(y_v),
    "test_samples": len(y_te),
}
with open(OUTPUT_DIR / "xgb_final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ─── Comparison ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("COMPARISON: No MIFS vs MIFS (XGBoost) vs MIFS (HistGB)")
print("=" * 70)

comparisons = [
    ("No MIFS (all 260K)", test_rmse, test_acc, X_tr.shape[1], best_params),
]

for folder, label in [("pipeline_output_xgboost", "MIFS + XGBoost"),
                       ("pipeline_output_histgb", "MIFS + HistGB")]:
    p = BASE_DIR / "data" / folder / "xgb_final_summary.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        comparisons.append((label, d["test_rmse"], d["test_acc"],
                           d.get("best_k", d.get("n_features", "?")),
                           d["best_params"]))

print(f"  {'Model':<25s} {'RMSE':>8s} {'ACC':>8s} {'Features':>10s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10}")
for name, rmse, acc, nf, params in comparisons:
    print(f"  {name:<25s} {rmse:>8.4f} {acc:>8.4f} {str(nf):>10s}")

print("=" * 70)
