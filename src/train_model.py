"""
src/train_model.py
------------------
Blue Carbon MRV — Sundarbans-Only Model Training
=================================================

Trains a RandomForestRegressor using ground-referenced field data
from the Sundarbans (Ganges-Brahmaputra Delta) region only.

Features : NDVI, lat, lon
Target   : carbon_tC_ha (total carbon density, tC/ha)

Outputs  : model.pkl (joblib) saved to the project root.

Usage:
    python src/train_model.py
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
# Works whether called from project root or src/
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(BASE_DIR, "..")
DATA_PATH  = os.path.join(ROOT_DIR, "data", "sundarbans_training_data_2023.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "model.pkl")

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("=" * 55)
print("  Blue Carbon MRV — Sundarbans Model Training")
print("=" * 55)

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at: {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"\n[DATA]  Loaded {len(df)} rows from: {os.path.basename(DATA_PATH)}")
print(f"        Columns: {list(df.columns)}")

# ── 2. Filter to Sundarbans region only ───────────────────────────────────────
if "region" in df.columns:
    before = len(df)
    df = df[df["region"].str.lower().str.strip() == "sundarbans"].copy()
    print(f"[DATA]  Filtered to Sundarbans: {before} → {len(df)} rows")

# ── 3. Define features & target ───────────────────────────────────────────────
FEATURE_COLS = ["NDVI", "lat", "lon"]
TARGET_COL   = "carbon_tC_ha"

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    print(f"[ERROR] Missing required columns: {missing}")
    sys.exit(1)

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

print(f"\n[FEATURES]  {FEATURE_COLS}")
print(f"[TARGET  ]  {TARGET_COL}")
print(f"[SHAPE   ]  X={X.shape}, y={y.shape}")
print(f"[TARGET  ]  min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f} tC/ha")

# ── 4. Train / Test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\n[SPLIT]  Train: {len(X_train)} | Test: {len(X_test)} (80/20)")

# ── 5. Train RandomForestRegressor ────────────────────────────────────────────
print("\n[TRAIN]  Fitting RandomForestRegressor ...")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)
print("[TRAIN]  Done.")

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 40)
print("  Evaluation (Test Set)")
print("=" * 40)
print(f"  R² Score : {r2:.4f}")
print(f"  MAE      : {mae:.4f} tC/ha")
print(f"  RMSE     : {rmse:.4f} tC/ha")
print("=" * 40)

# ── 7. Feature importances ────────────────────────────────────────────────────
print("\n[FEATURES]  Importance:")
for name, imp in zip(FEATURE_COLS, model.feature_importances_):
    print(f"  {name:<8} : {imp:.4f}")

# ── 8. Save model ─────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
print(f"\n[SAVED]  model.pkl → {MODEL_PATH}")
print("[DONE]   Model ready for inference via backend/app.py\n")
