"""
train.py
--------
Step 2 of the Blue Carbon MRV Pipeline.

Trains a Random Forest Regressor to predict mangrove carbon stock
(tC/ha) from six Sentinel-2 features: B2, B3, B4, B8, B11, NDVI.

Outputs
-------
  outputs/evaluation_report.txt   – R², RMSE, MAE, cross-val summary,
                                    and feature importance ranking
  outputs/feature_importance.png  – bar chart of feature importances
  outputs/actual_vs_predicted.png – scatter of test-set predictions
  models/carbon_model.pkl         – saved Random Forest model

Run:
    python train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "mangrove_carbon_samples.csv")
MODEL_DIR   = "models"
OUTPUT_DIR  = "outputs"
MODEL_PATH  = os.path.join(MODEL_DIR, "carbon_model.pkl")
REPORT_PATH = os.path.join(OUTPUT_DIR, "evaluation_report.txt")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature columns used by the model (order matters — must match predictor.py)
FEATURE_COLS = ["B2_blue", "B3_green", "B4_red", "B8_nir", "B11_swir", "NDVI"]
TARGET_COL   = "carbon_stock_tC_ha"

RANDOM_STATE = 42   # fixed seed → reproducible results


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  Blue Carbon MRV — Model Training")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
X  = df[FEATURE_COLS].values
y  = df[TARGET_COL].values

print(f"\n✔  Loaded {len(df)} samples | features: {FEATURE_COLS}")


# ── 2. Train / test split (80 / 20) ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print(f"   Train size: {len(X_train)} | Test size: {len(X_test)}")


# ── 3. Build and fit the Random Forest ────────────────────────────────────────
# n_estimators=200: enough trees for stability on a small dataset
# max_depth=None:   let trees grow fully (avoids underfitting on 20 samples)
# min_samples_split=2, min_samples_leaf=1: keep the model flexible
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
)
rf.fit(X_train, y_train)
print("\n✔  Random Forest trained.")


# ── 4. Test-set evaluation ─────────────────────────────────────────────────────
y_pred = rf.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)


# ── 5. 5-fold Cross-Validation ────────────────────────────────────────────────
# cross_validate returns per-fold scores; we report the mean ± std
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = cross_validate(
    rf, X, y,
    cv=kf,
    scoring=["r2", "neg_root_mean_squared_error"],
    return_train_score=False,
)

cv_r2_mean   = cv_results["test_r2"].mean()
cv_r2_std    = cv_results["test_r2"].std()
cv_rmse_mean = (-cv_results["test_neg_root_mean_squared_error"]).mean()
cv_rmse_std  = (-cv_results["test_neg_root_mean_squared_error"]).std()


# ── 6. Feature importance ─────────────────────────────────────────────────────
importances = rf.feature_importances_
fi_series   = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)


# ── 7. Build evaluation report ────────────────────────────────────────────────
report_lines = [
    "=" * 60,
    "  Blue Carbon MRV - Model Evaluation Report",
    "=" * 60,
    "",
    "-- Test-Set Metrics (80/20 split) " + "-" * 26,
    f"  R2   (coefficient of determination): {r2:.4f}",
    f"  RMSE (root mean squared error)      : {rmse:.4f} tC/ha",
    f"  MAE  (mean absolute error)           : {mae:.4f} tC/ha",
    "",
    "-- 5-Fold Cross-Validation " + "-" * 33,
    f"  Mean R2  : {cv_r2_mean:.4f}  +/- {cv_r2_std:.4f}",
    f"  Mean RMSE: {cv_rmse_mean:.4f} +/- {cv_rmse_std:.4f} tC/ha",
    "",
    "-- Feature Importance Ranking " + "-" * 30,
]
for i, (feat, imp) in enumerate(fi_series.items(), 1):
    bar = "█" * int(imp * 40)
    report_lines.append(f"  #{i:>2}  {feat:<12}: {imp:.4f}  {bar}")

report_lines += [
    "",
    "-- Model Configuration " + "-" * 37,
    f"  Algorithm   : Random Forest Regressor",
    f"  n_estimators: 200",
    f"  max_depth   : None (fully grown trees)",
    f"  Random seed : {RANDOM_STATE}",
    f"  Train/test  : 80/20",
    "=" * 60,
]
report_text = "\n".join(report_lines)

# Print to console
print()
print(report_text)

# Save to file (utf-8 ensures box-drawing and special chars survive on Windows)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\n✔  Report saved → {REPORT_PATH}")


# ── 8. Save the trained model ─────────────────────────────────────────────────
joblib.dump(rf, MODEL_PATH)
print(f"✔  Model saved  → {MODEL_PATH}")


# ── 9. Plots ──────────────────────────────────────────────────────────────────
DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
ACCENT    = "#4ade80"   # green — mangrove theme
TEXT      = "#cccccc"
GRID_CLR  = "#2a2d3e"

# -- 9a. Feature importance bar chart -----------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(PANEL_BG)

colors = [ACCENT if v == fi_series.max() else "#6ee7b7" for v in fi_series.values]
bars = ax.barh(fi_series.index[::-1], fi_series.values[::-1],
               color=colors[::-1], edgecolor="#ffffff22", linewidth=0.5)

ax.set_xlabel("Relative Importance", color=TEXT, fontsize=11)
ax.set_title("Random Forest — Feature Importance\n(Mangrove Carbon Stock Prediction)",
             color="#ffffff", fontsize=12, pad=12)
ax.tick_params(colors=TEXT)
ax.grid(axis="x", color=GRID_CLR, linestyle="--", linewidth=0.5)
for spine in ax.spines.values():
    spine.set_edgecolor("#333344")

# Value labels on bars
for bar, val in zip(bars, fi_series.values[::-1]):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", color=TEXT, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150,
            facecolor=DARK_BG)
plt.close()
print(f"✔  Feature importance plot saved → {os.path.join(OUTPUT_DIR, 'feature_importance.png')}")


# -- 9b. Actual vs Predicted scatter ------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(PANEL_BG)

ax.scatter(y_test, y_pred, color=ACCENT, edgecolors="#ffffff44",
           s=110, zorder=3, label="Predictions")

# Perfect-prediction diagonal
lo = min(y_test.min(), y_pred.min()) - 2
hi = max(y_test.max(), y_pred.max()) + 2
ax.plot([lo, hi], [lo, hi], color="#ff6b6b", linestyle="--",
        linewidth=1.5, label="Perfect fit", zorder=2)

ax.set_xlabel("Actual Carbon Stock (tC/ha)", color=TEXT, fontsize=11)
ax.set_ylabel("Predicted Carbon Stock (tC/ha)", color=TEXT, fontsize=11)
ax.set_title(f"Actual vs Predicted — Test Set\n(R² = {r2:.3f}, RMSE = {rmse:.2f} tC/ha)",
             color="#ffffff", fontsize=12, pad=12)
ax.legend(facecolor=PANEL_BG, labelcolor=TEXT, edgecolor="#333344")
ax.tick_params(colors=TEXT)
ax.grid(color=GRID_CLR, linestyle="--", linewidth=0.5)
for spine in ax.spines.values():
    spine.set_edgecolor("#333344")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150,
            facecolor=DARK_BG)
plt.close()
print(f"✔  Actual vs predicted plot saved → {os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png')}")

print("\nTraining complete.\n")
