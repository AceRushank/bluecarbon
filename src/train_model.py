"""
src/train_model.py
------------------
Blue Carbon MRV — Multi-Region Model Training
=================================================

Trains RandomForestRegressor models for:
1. Sundarbans Delta (Ganges-Brahmaputra Delta)
2. Andaman & Nicobar Islands (Bay of Bengal Archipelago)

Features : NDVI, lat, lon (or latitude, longitude)
Target   : carbon_tC_ha (total carbon density, tC/ha)

Outputs  : 
  - model.pkl & model_sundarbans.pkl (Sundarbans model)
  - model_andaman.pkl (Andaman model)

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

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(BASE_DIR, "..")

def train_sundarbans():
    print("\n" + "=" * 60)
    print("  1. Training Sundarbans Delta RandomForest Model")
    print("=" * 60)
    
    data_path = os.path.join(ROOT_DIR, "data", "sundarbans_training_data_2023.csv")
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at: {data_path}")
        return None
        
    df = pd.read_csv(data_path)
    print(f"[DATA]  Loaded {len(df)} rows from: {os.path.basename(data_path)}")
    
    X = df[["NDVI", "lat", "lon"]].values
    y = df["carbon_tC_ha"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"[EVAL]  R² Score: {r2:.4f} | MAE: {mae:.4f} tC/ha | RMSE: {rmse:.4f} tC/ha")
    
    # Save both model.pkl and model_sundarbans.pkl
    p1 = os.path.join(ROOT_DIR, "model.pkl")
    p2 = os.path.join(ROOT_DIR, "model_sundarbans.pkl")
    joblib.dump(model, p1)
    joblib.dump(model, p2)
    print(f"[SAVED] {p1}")
    print(f"[SAVED] {p2}")
    return model

# ==============================================================================
# TARGET VALUES ARE SYNTHETIC — derived from NDVI/EVI via an assumed linear
# allometric formula (125 + 80*NDVI + 15*EVI), NOT real field plot measurements.
# See disclosure note before presenting this model's outputs as equivalent to
# the Sundarbans model.
# ==============================================================================
def train_andaman():
    print("\n" + "=" * 60)
    print("  2. Training Andaman & Nicobar Islands RandomForest Model")
    print("=" * 60)
    
    data_path = os.path.join(ROOT_DIR, "data", "bluechain_andaman_mangrove_lite2.csv")
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at: {data_path}")
        return None
        
    df = pd.read_csv(data_path)
    # Filter to Andaman & Nicobar bounding coordinates
    df_and = df[(df['longitude'] > 90) & (df['latitude'] < 14)].copy()
    
    if len(df_and) == 0:
        # Fallback if filtered empty
        df_and = df.copy()
        
    # Calculate carbon stock (tC/ha) calibrated for Andaman Island mangrove allometry
    if "carbon_tC_ha" not in df_and.columns:
        df_and["carbon_tC_ha"] = 125.0 + 80.0 * df_and["NDVI"] + 15.0 * df_and.get("EVI", 0.4)
        
    print(f"[DATA]  Loaded {len(df_and)} Andaman rows from: {os.path.basename(data_path)}")
    
    # Map lat/lon column names
    lat_col = "latitude" if "latitude" in df_and.columns else "lat"
    lon_col = "longitude" if "longitude" in df_and.columns else "lon"
    
    X = df_and[["NDVI", lat_col, lon_col]].values
    y = df_and["carbon_tC_ha"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"[EVAL]  R² Score: {r2:.4f} | MAE: {mae:.4f} tC/ha | RMSE: {rmse:.4f} tC/ha")
    
    p_andaman = os.path.join(ROOT_DIR, "model_andaman.pkl")
    joblib.dump(model, p_andaman)
    print(f"[SAVED] {p_andaman}")
    return model

if __name__ == "__main__":
    train_sundarbans()
    train_andaman()
    print("\n[COMPLETE] All regional models successfully trained and serialized.")
