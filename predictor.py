"""
predictor.py
------------
Step 3 of the Blue Carbon MRV Pipeline.

Provides a clean, reusable function `predict_carbon(bands)` that:
  1. Loads the saved Random Forest model from disk (once).
  2. Accepts a dictionary of Sentinel-2 band reflectance values.
  3. Returns a single float: the estimated carbon stock in tC/ha.

Usage example
-------------
    from predictor import predict_carbon

    result = predict_carbon({
        "B2_blue":   0.045,
        "B3_green":  0.068,
        "B4_red":    0.052,
        "B8_nir":    0.312,
        "B11_swir":  0.098,
        "NDVI":      0.714,
    })
    print(f"Estimated carbon stock: {result:.2f} tC/ha")

Run directly to execute the built-in sanity-check test:
    python predictor.py
"""

import os
import joblib
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", "carbon_model.pkl")
DATA_PATH    = os.path.join("data",   "mangrove_carbon_samples.csv")

# Feature order MUST match the order used during training (train.py).
# Changing this order would silently corrupt predictions.
FEATURE_COLS = ["B2_blue", "B3_green", "B4_red", "B8_nir", "B11_swir", "NDVI"]

# Module-level cache: the model is loaded once on first call, then reused.
_model = None


def _load_model():
    """Load the Random Forest model from disk (cached after first call)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Please run `python train.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_carbon(bands: dict) -> float:
    """
    Predict mangrove carbon stock from Sentinel-2 band reflectances.

    Parameters
    ----------
    bands : dict
        Dictionary mapping feature names to their float reflectance values.
        Required keys: 'B2_blue', 'B3_green', 'B4_red',
                       'B8_nir', 'B11_swir', 'NDVI'.

    Returns
    -------
    float
        Estimated carbon stock in tonnes of carbon per hectare (tC/ha).

    Raises
    ------
    ValueError
        If any required feature key is missing from `bands`.
    FileNotFoundError
        If the model file has not been created yet (run train.py first).
    """
    # Validate that all required features are present
    missing = [f for f in FEATURE_COLS if f not in bands]
    if missing:
        raise ValueError(
            f"Missing features in input dict: {missing}\n"
            f"Required: {FEATURE_COLS}"
        )

    # Build a 2-D array with shape (1, n_features) — sklearn expects this
    feature_vector = np.array([[bands[f] for f in FEATURE_COLS]])

    model = _load_model()
    prediction = model.predict(feature_vector)

    # prediction is a numpy array of length 1; return a plain Python float
    return float(prediction[0])


# ── Built-in sanity-check test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  predictor.py — Sanity Check")
    print("=" * 60)

    # ── Test 1: hard-coded example from the task spec ──────────────────────────
    print("\nTest 1 — Hard-coded example (site S1):")
    sample_bands = {
        "B2_blue":  0.045,
        "B3_green": 0.068,
        "B4_red":   0.052,
        "B8_nir":   0.312,
        "B11_swir": 0.098,
        "NDVI":     0.714,
    }
    pred = predict_carbon(sample_bands)
    actual_s1 = 52.3   # known value for S1 from the CSV
    print(f"  Input   : {sample_bands}")
    print(f"  Actual  : {actual_s1:.2f} tC/ha")
    print(f"  Predicted: {pred:.2f} tC/ha")
    print(f"  Error   : {abs(pred - actual_s1):.2f} tC/ha")

    # ── Test 2: loop over every sample in the CSV and compare ──────────────────
    print("\nTest 2 — All 20 samples from the CSV:")
    df = pd.read_csv(DATA_PATH)
    print(f"  {'Site':<6} {'Actual':>12} {'Predicted':>12} {'|Error|':>10}")
    print("  " + "-" * 44)
    errors = []
    for _, row in df.iterrows():
        band_vals = {f: row[f] for f in FEATURE_COLS}
        y_pred    = predict_carbon(band_vals)
        y_true    = row["carbon_stock_tC_ha"]
        err       = abs(y_pred - y_true)
        errors.append(err)
        print(f"  {row['site_id']:<6} {y_true:>10.2f}  {y_pred:>10.2f}  {err:>9.2f}")

    mean_err = np.mean(errors)
    print(f"\n  Mean absolute error across all samples: {mean_err:.2f} tC/ha")
    print("\n✔  Sanity check complete.\n")
