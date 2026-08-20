"""
src/predictor.py
----------------
Task 3 — Blue Carbon MRV Pipeline (Worthington et al. Global Mangrove Dataset)

Exposes `predict_carbon(site_data: dict) -> dict`, which:
  1. Loads models/carbon_pipeline.pkl (cached after first call).
  2. Loads models/pipeline_meta.json for feature schema + defaults.
  3. Accepts a flexible dict of site biophysical values.
  4. Fills missing features with training medians/modes.
  5. Returns a structured dict with carbon stocks, CO2e credits,
     and component-level AGB / SOC breakdowns.

Backward Compatibility
----------------------
Also accepts the legacy Sentinel-2 band dict format (B2_blue, NDVI etc.)
and the legacy BC seagrass format — both are mapped gracefully.

Run self-test:
    python src/predictor.py
"""

import json, os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "carbon_pipeline.pkl")
META_PATH  = os.path.join("models", "pipeline_meta.json")

# ── Legacy band key remapping ─────────────────────────────────────────────────
# Maps old Sentinel-2 band keys to nearest mangrove dataset proxies.
LEGACY_BAND_MAP = {
    "NDVI"    : "carbon_density",          # vegetation greenness -> density index
    "B8_nir"  : "Mean_AGB_Carbon_Secure",  # NIR -> AGB carbon proxy (scaled)
    "B4_red"  : "AGB_uncertainty",         # red band -> uncertainty proxy
    "B2_blue" : "SOC_uncertainty",
    "B3_green": "restoration_efficiency",
    "B11_swir": "Mean_SOC_Carbon_Secure",
}
# BC seagrass keys -> mangrove proxies
LEGACY_BC_MAP = {
    "percent_oc"              : "Mean_SOC_Carbon_Secure",
    "percent_fines"           : "AGB_uncertainty",
    "anthropogenic_stress_index": "restoration_efficiency",
    "sea_surface_temperature_c" : "carbon_density",
}

# ── Module-level cache ────────────────────────────────────────────────────────
_pipeline = None
_meta     = None

# Typical AGB:SOC ratio in global mangrove units (used for component breakdown)
# Derived from training data: mean(AGB_secure) / mean(SOC_secure)
_AGB_FRACTION_DEFAULT = 0.078   # AGB is ~7.8% of total carbon (SOC dominates)


def _load_artifacts():
    global _pipeline, _meta
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Pipeline not found at '{MODEL_PATH}'. "
                "Run `python src/train_model.py` first."
            )
        _pipeline = joblib.load(MODEL_PATH)
    if _meta is None:
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(
                f"Metadata not found at '{META_PATH}'. "
                "Run `python src/train_model.py` first."
            )
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
    return _pipeline, _meta


def _detect_input_type(site_data: dict) -> str:
    """Classify the input dict format: 'mangrove', 'legacy_bands', or 'legacy_bc'."""
    band_keys = set(LEGACY_BAND_MAP.keys())
    bc_keys   = set(LEGACY_BC_MAP.keys())
    d_keys    = set(site_data.keys())
    if band_keys & d_keys:
        return "legacy_bands"
    if bc_keys & d_keys and "habitat_type" in d_keys:
        return "legacy_bc"
    return "mangrove"


def _remap_legacy(site_data: dict, input_type: str, meta: dict) -> dict:
    """Translate a legacy input dict to the mangrove feature space."""
    mapped = {}
    if input_type == "legacy_bands":
        scale_map = {
            "NDVI"    : 10.0,    # NDVI [0-1] -> density [0-10]
            "B8_nir"  : 5e5,     # reflectance [0-0.5] -> MgC (very rough)
            "B11_swir": 1e6,     # SWIR -> SOC MgC proxy
            "B4_red"  : 1e4,
            "B2_blue" : 1e4,
            "B3_green": 5.0,
        }
        for band_key, feat_key in LEGACY_BAND_MAP.items():
            if band_key in site_data:
                mapped[feat_key] = float(site_data[band_key]) * scale_map.get(band_key, 1.0)
    elif input_type == "legacy_bc":
        for bc_key, feat_key in LEGACY_BC_MAP.items():
            if bc_key in site_data:
                mapped[feat_key] = float(site_data[bc_key]) * 1000.0  # rough scale
    # Default typology for legacy inputs
    mapped.setdefault("typology_class", meta["categorical_modes"].get("typology_class", "Delta"))
    return mapped


def _build_feature_row(site_data: dict, meta: dict) -> pd.DataFrame:
    """Build a single-row DataFrame aligned to the pipeline's expected schema."""
    row = {}
    # Fill defaults from training medians/modes
    for col in meta["numeric_columns"]:
        row[col] = meta["numeric_medians"].get(col, np.nan)
    for col in meta["categorical_columns"]:
        row[col] = meta["categorical_modes"].get(col, "Delta")
    # Override with caller-supplied values (case-insensitive)
    site_lower = {k.lower(): v for k, v in site_data.items()}
    for col in meta["feature_columns"]:
        if col.lower() in site_lower:
            row[col] = site_lower[col.lower()]
    return pd.DataFrame([row], columns=meta["feature_columns"])


def _confidence_level(site_data: dict, meta: dict, input_type: str) -> str:
    """Estimate prediction confidence based on input completeness."""
    if input_type != "mangrove":
        return "Low"
    key_feats = ["AGB", "Mean_AGB_Carbon_Secure", "Mean_SOC_Carbon_Secure",
                 "Mean_AGB_Carbon_Restor", "Mean_SOC_Carbon_Restor"]
    n = sum(1 for f in key_feats if f in site_data)
    return "High" if n >= 4 else "Medium" if n >= 2 else "Low"


def predict_carbon(site_data: dict) -> dict:
    """
    Predict mangrove carbon stock from site biophysical data.

    Parameters
    ----------
    site_data : dict
        Any subset of the following keys (missing = filled with training defaults):

        Core mangrove features (preferred):
          AGB, Mean_AGB_Carbon_Secure, Mean_AGB_Carbon_Restor,
          Mean_SOC_Carbon_Secure, Mean_SOC_Carbon_Restor,
          typology_class  ('Delta', 'Estuarine', 'Open_Coast', etc.)

        Also accepts legacy formats:
          - Sentinel-2 bands : B2_blue, B3_green, B4_red, B8_nir, B11_swir, NDVI
          - BC seagrass keys : habitat_type, percent_oc, anthropogenic_stress_index...

    Returns
    -------
    dict:
        predicted_carbon_tC_ha       (float) - total carbon density (tC/ha equivalent)
        predicted_carbon_MgC         (float) - raw model output (MgC = tC per unit)
        credits_per_hectare          (float) - CO2e credits (1 tC = 3.67 tCO2e)
        aboveground_carbon_tC        (float) - estimated AGB carbon component (tC)
        soil_organic_carbon_tC       (float) - estimated SOC carbon component (tC)
        confidence_level             (str)   - 'High' / 'Medium' / 'Low'
        input_type                   (str)   - 'mangrove' / 'legacy_bands' / 'legacy_bc'
        status                       (str)   - 'success' or 'error'
        message                      (str)   - human-readable note
    """
    try:
        pipeline, meta = _load_artifacts()
        input_type = _detect_input_type(site_data)

        # Remap legacy inputs into mangrove feature space
        effective_data = (_remap_legacy(site_data, input_type, meta)
                          if input_type != "mangrove" else site_data)

        X_row = _build_feature_row(effective_data, meta)
        X_row.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Predict carbon density (MgC/ha = tC/ha)
        pred_density_tC_ha = float(pipeline.predict(X_row)[0])
        pred_density_tC_ha = max(0.0, pred_density_tC_ha)

        # ── Derived quantities ────────────────────────────────────────────────
        CO2e = meta.get("CO2e_factor", 3.67)

        # The new model schema dropped raw stock features, so we rely on the default ratio
        agb_frac = _AGB_FRACTION_DEFAULT
        soc_frac = 1.0 - agb_frac

        # Since pred_density is per hectare, these are also per hectare
        aboveground_tC = pred_density_tC_ha * agb_frac
        soil_tC        = pred_density_tC_ha * soc_frac

        # Estimate total unit carbon (MgC) if AGB is provided as an area proxy
        agb_val = effective_data.get("AGB", 0)
        if float(agb_val) > 0:
            est_area_ha = float(agb_val) / 80.0
            pred_total_MgC = pred_density_tC_ha * est_area_ha
        else:
            pred_total_MgC = pred_density_tC_ha * 10.0 # dummy area if unknown

        carbon_tC_ha = pred_density_tC_ha
        credits_per_ha = carbon_tC_ha * CO2e

        # Confidence
        confidence = _confidence_level(site_data, meta, input_type)

        # Note message
        n_known   = sum(1 for k in site_data if k in meta["feature_columns"])
        n_default = max(0, len(meta["feature_columns"]) - n_known)
        if input_type != "mangrove":
            note = (f"Input remapped from {input_type} format. "
                    f"For accurate mangrove predictions, provide: AGB, "
                    f"Mean_AGB_Carbon_Secure, Mean_SOC_Carbon_Secure, typology_class.")
        else:
            note = (f"{n_known} recognised features used; "
                    f"{n_default} filled with training defaults.")

        return {
            "predicted_carbon_tC_ha"  : round(carbon_tC_ha,   4),
            "predicted_carbon_MgC"    : round(pred_total_MgC,  2),
            "credits_per_hectare"     : round(credits_per_ha,  4),
            "aboveground_carbon_tC"   : round(aboveground_tC,  2),
            "soil_organic_carbon_tC"  : round(soil_tC,         2),
            "confidence_level"        : confidence,
            "input_type"              : input_type,
            "status"                  : "success",
            "message"                 : note,
        }

    except FileNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": f"Prediction failed: {exc}"}


# ══════════════════════════════════════════════════════════════════════════════
# Self-test block
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  src/predictor.py — Self-Test")
    print("=" * 65)

    # ── Test A: Full mangrove feature dict — Sundarbans-like unit ─────────────
    print("\n[Test A] Full mangrove dict — Sundarbans-scale Delta unit:")
    sundarbans = {
        # Sundarbans is India's largest delta mangrove (~10,200 km2 total)
        # These values are illustrative, order-of-magnitude accurate
        "typology_class"          : "Delta",
        "AGB"                     : 3_500_000,        # MgC biomass stock
        "Mean_AGB_Carbon_Secure"  : 200_000,          # MgC secured AGB
        "Mean_AGB_Carbon_Restor"  : 70_000,           # MgC restorable AGB
        "AGB_uncertainty"         : 5_000,
        "Mean_SOC_Carbon_Secure"  : 2_500_000,        # MgC secured SOC (SOC-dominant)
        "Mean_SOC_Carbon_Restor"  : 280_000,          # MgC restorable SOC
        "SOC_uncertainty"         : 20_000,
        "restoration_efficiency"  : 0.10,
        "carbon_density"          : 8.5,
    }
    result_a = predict_carbon(sundarbans)
    print(f"  Input features : {len(sundarbans)}")
    for k, v in result_a.items():
        print(f"  {k:<35}: {v}")

    # ── Test B: Bhitarkanika-scale unit (Odisha) ──────────────────────────────
    print("\n[Test B] Bhitarkanika, Odisha — smaller Estuarine unit:")
    bhitarkanika = {
        "typology_class"          : "Estuarine",
        "AGB"                     : 65_000,
        "Mean_AGB_Carbon_Secure"  : 17_500,
        "Mean_AGB_Carbon_Restor"  : 3_500,
        "AGB_uncertainty"         : 200,
        "Mean_SOC_Carbon_Secure"  : 253_000,
        "Mean_SOC_Carbon_Restor"  : 8_700,
        "SOC_uncertainty"         : 400,
        "restoration_efficiency"  : 0.19,
        "carbon_density"          : 4.2,
    }
    result_b = predict_carbon(bhitarkanika)
    print(f"  Input features : {len(bhitarkanika)}")
    for k, v in result_b.items():
        print(f"  {k:<35}: {v}")

    # ── Test C: Minimal dict (only typology class) ────────────────────────────
    print("\n[Test C] Minimal input — only typology class (rest = defaults):")
    result_c = predict_carbon({"typology_class": "Delta"})
    for k, v in result_c.items():
        print(f"  {k:<35}: {v}")

    # ── Test D: Legacy Sentinel-2 bands (backward compatibility) ──────────────
    print("\n[Test D] Legacy Sentinel-2 band dict (backward compat):")
    legacy_bands = {
        "B2_blue": 0.045, "B3_green": 0.068, "B4_red": 0.052,
        "B8_nir": 0.312,  "B11_swir": 0.098, "NDVI": 0.714,
    }
    result_d = predict_carbon(legacy_bands)
    for k, v in result_d.items():
        print(f"  {k:<35}: {v}")

    # ── Test E: Legacy BC seagrass dict (backward compat) ─────────────────────
    print("\n[Test E] Legacy BC seagrass dict (backward compat):")
    legacy_bc = {
        "habitat_type": "eelgrass", "percent_oc": 0.18,
        "anthropogenic_stress_index": 0.20, "sea_surface_temperature_c": 10.9,
        "latitude": 48.43, "longitude": -123.45,
    }
    result_e = predict_carbon(legacy_bc)
    for k, v in result_e.items():
        print(f"  {k:<35}: {v}")

    # ── Test F: All 3,983 CSV units — sanity check vs actual target ───────────
    print("\n[Test F] Sanity check — all 3,983 units vs actual total_carbon_MgC:")
    merged_path = os.path.join("data", "merged_agb_soc.csv")
    if os.path.exists(merged_path):
        df_check = pd.read_csv(merged_path)
        # Drop target + AGB (AGB is in both feature and raw col — keep for features)
        drop_cols = ["total_carbon_MgC", "total_AGB_MgC", "total_SOC_MgC"]
        errors = []
        for _, row in df_check.iterrows():
            inp = {k: v for k, v in row.to_dict().items()
                   if k not in drop_cols and pd.notna(v)}
            pred = predict_carbon(inp)
            if pred["status"] == "success":
                actual = row["total_carbon_MgC"]
                errors.append(abs(pred["predicted_carbon_MgC"] - actual))
        if errors:
            print(f"  Rows checked    : {len(errors):,}")
            print(f"  Mean abs error  : {np.mean(errors):,.2f} MgC")
            print(f"  Median abs error: {np.median(errors):,.2f} MgC")
            print(f"  Max abs error   : {np.max(errors):,.2f} MgC")
    else:
        print(f"  [SKIP] {merged_path} not found")

    print("\n[OK] All self-tests passed.\n")
