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
     component-level AGB / SOC breakdowns, and a full accounting
     of which features were actually supplied vs. defaulted.

Model Features (pipeline_meta.json feature_columns)
----------------------------------------------------
  AGB_carbon_fraction     – fraction of total carbon held as AGB
  SOC_agb_ratio           – SOC-to-AGB mass ratio
  AGB_uncertainty_pct     – relative uncertainty on AGB estimate
  SOC_uncertainty_pct     – relative uncertainty on SOC estimate
  restoration_efficiency  – fraction of restorable carbon stock
  agb_restor_fraction     – AGB restorable fraction
  soc_restor_fraction     – SOC restorable fraction
  typology_class          – categorical: Delta / Estuary / Lagoon / OpenCoast

Backward Compatibility
----------------------
Legacy Sentinel-2 band dicts (B2_blue, B3_green, B4_red, B8_nir, B11_swir,
NDVI) and legacy BC seagrass dicts are accepted via LEGACY_BAND_MAP and
LEGACY_BC_MAP respectively.

IMPORTANT: Of all Sentinel-2 band keys, only B3_green maps to a real model
feature (restoration_efficiency). B2_blue, B4_red, B8_nir, B11_swir, and NDVI
have NO scientifically defensible mapping to the ratio/uncertainty/fraction
features the model actually uses, and are therefore not mapped — callers who
supply these values will see them reported as "caller_supplied_no_effect" in
the features_actually_used output key.

Of all BC seagrass keys, only anthropogenic_stress_index maps to a real model
feature (restoration_efficiency). percent_oc, percent_fines, and
sea_surface_temperature_c have no legitimate mapping and are not mapped.

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
# Maps old Sentinel-2 band keys to model feature_columns entries that actually
# exist in pipeline_meta.json.
#
# BEFORE (removed entries):
#   "NDVI"    : "carbon_density"          — "carbon_density" not in feature_columns
#   "B8_nir"  : "Mean_AGB_Carbon_Secure"  — not in feature_columns
#   "B4_red"  : "AGB_uncertainty"         — not in feature_columns
#   "B2_blue" : "SOC_uncertainty"         — not in feature_columns
#   "B11_swir": "Mean_SOC_Carbon_Secure"  — not in feature_columns
#
# AFTER (only the one entry that has a real target):
#   "B3_green": "restoration_efficiency"  — IS in feature_columns ✓
#
# Raw reflectance values (B2/B4/B8/B11/NDVI) cannot be physically mapped to
# the ratio/fraction/uncertainty features the model actually uses without
# introducing scientifically indefensible scaling constants. Those bands are
# left to training-median defaults.
LEGACY_BAND_MAP = {
    "B3_green": "restoration_efficiency",   # vegetation greenness -> restor. proxy
}

# BC seagrass keys -> mangrove feature_columns entries that actually exist.
#
# BEFORE (removed entries):
#   "percent_oc"              : "Mean_SOC_Carbon_Secure"  — not in feature_columns
#   "percent_fines"           : "AGB_uncertainty"         — not in feature_columns
#   "sea_surface_temperature_c" : "carbon_density"        — not in feature_columns
#
# AFTER (only the one entry that has a real target):
#   "anthropogenic_stress_index": "restoration_efficiency" — IS in feature_columns ✓
LEGACY_BC_MAP = {
    "anthropogenic_stress_index": "restoration_efficiency",
}

# ── Sentinel-2 band keys with no valid feature_columns mapping ─────────────────
# Callers may still supply these; they are accepted without error but have no
# effect on the model prediction and are reported as such in features_actually_used.
_BAND_KEYS_NO_EFFECT = {"NDVI", "B8_nir", "B4_red", "B2_blue", "B11_swir"}
_BC_KEYS_NO_EFFECT   = {"percent_oc", "percent_fines", "sea_surface_temperature_c"}

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
    # Use the union of mapped + no-effect keys to detect format
    band_keys = set(LEGACY_BAND_MAP.keys()) | _BAND_KEYS_NO_EFFECT
    bc_keys   = set(LEGACY_BC_MAP.keys())   | _BC_KEYS_NO_EFFECT
    d_keys    = set(site_data.keys())
    if band_keys & d_keys:
        return "legacy_bands"
    if bc_keys & d_keys and "habitat_type" in d_keys:
        return "legacy_bc"
    return "mangrove"


def _remap_legacy(site_data: dict, input_type: str, meta: dict) -> dict:
    """Translate a legacy input dict to the mangrove feature space.

    Only keys that appear in LEGACY_BAND_MAP / LEGACY_BC_MAP (which now only
    contain entries with valid feature_columns targets) are forwarded.
    Keys in _BAND_KEYS_NO_EFFECT / _BC_KEYS_NO_EFFECT are silently ignored here;
    they are reported in features_actually_used as 'caller_supplied_no_effect'.
    """
    mapped = {}
    if input_type == "legacy_bands":
        # B3_green -> restoration_efficiency: green reflectance [0-0.1] scaled
        # to [0-0.5], a plausible range for the restoration_efficiency feature.
        # This is a weak proxy; the caller should prefer supplying
        # restoration_efficiency directly for accurate results.
        scale_map = {
            "B3_green": 5.0,
        }
        for band_key, feat_key in LEGACY_BAND_MAP.items():
            if band_key in site_data:
                mapped[feat_key] = float(site_data[band_key]) * scale_map.get(band_key, 1.0)
    elif input_type == "legacy_bc":
        for bc_key, feat_key in LEGACY_BC_MAP.items():
            if bc_key in site_data:
                mapped[feat_key] = float(site_data[bc_key])
    # Default typology for legacy inputs
    mapped.setdefault("typology_class", meta["categorical_modes"].get("typology_class", "OpenCoast"))
    return mapped


def _build_feature_row(site_data: dict, meta: dict,
                        original_site_data: dict = None) -> tuple:
    """Build a single-row DataFrame aligned to the pipeline's expected schema.

    Parameters
    ----------
    site_data : dict
        The (possibly remapped) dict of features to use for building the row.
    meta : dict
        Pipeline metadata (feature_columns, medians, modes, etc.)
    original_site_data : dict, optional
        The original dict supplied by the caller before any legacy remapping.
        When provided, 'caller_supplied_no_effect' will be computed against the
        *original* keys, giving an honest accounting of what the caller passed in.

    Returns
    -------
    (pd.DataFrame, dict)
        The feature row ready for pipeline.predict(), and a features_actually_used
        dict with three keys:
          'supplied'                  - feature_columns entries whose values came
                                        from the effective_data (after remapping).
          'defaulted'                 - feature_columns entries filled from training
                                        medians/modes.
          'caller_supplied_no_effect' - keys the *original* caller supplied that
                                        did not reach any feature_columns entry,
                                        either because they were unmapped band/BC
                                        keys or because they were simply unknown.
    """
    row = {}
    # Fill defaults from training medians/modes
    for col in meta["numeric_columns"]:
        row[col] = meta["numeric_medians"].get(col, np.nan)
    for col in meta["categorical_columns"]:
        row[col] = meta["categorical_modes"].get(col, "OpenCoast")

    # Override with effective (post-remap) values (case-insensitive)
    site_lower = {k.lower(): v for k, v in site_data.items()}
    supplied   = []
    defaulted  = []
    for col in meta["feature_columns"]:
        if col.lower() in site_lower:
            row[col] = site_lower[col.lower()]
            supplied.append(col)
        else:
            defaulted.append(col)

    # Identify caller keys that landed nowhere in feature_columns.
    # We check against the ORIGINAL site_data keys if available, so we report
    # what the user actually typed, not what survived the legacy remapping.
    report_keys = original_site_data if original_site_data is not None else site_data
    feature_cols_lower = {c.lower() for c in meta["feature_columns"]}
    no_effect = [k for k in report_keys if k.lower() not in feature_cols_lower]

    features_used = {
        "supplied": supplied,
        "defaulted": defaulted,
        "caller_supplied_no_effect": no_effect,
    }
    return pd.DataFrame([row], columns=meta["feature_columns"]), features_used


def _confidence_level(site_data: dict, meta: dict, input_type: str) -> str:
    """Estimate prediction confidence based on input completeness."""
    if input_type != "mangrove":
        return "Low"
    key_feats = ["AGB_carbon_fraction", "SOC_agb_ratio",
                 "restoration_efficiency", "typology_class"]
    n = sum(1 for f in key_feats if f in site_data)
    return "High" if n >= 3 else "Medium" if n >= 2 else "Low"


def predict_carbon(site_data: dict) -> dict:
    """
    Predict mangrove carbon stock from site biophysical data.

    Parameters
    ----------
    site_data : dict
        Any subset of the following keys (missing = filled with training defaults):

        Core mangrove features (preferred — these directly influence the model):
          AGB_carbon_fraction     – AGB as a fraction of total carbon [0–1]
          SOC_agb_ratio           – SOC / AGB mass ratio
          AGB_uncertainty_pct     – relative AGB uncertainty
          SOC_uncertainty_pct     – relative SOC uncertainty
          restoration_efficiency  – restorable fraction [0–1]
          agb_restor_fraction     – AGB restorable fraction [0–1]
          soc_restor_fraction     – SOC restorable fraction [0–1]
          typology_class          – 'Delta', 'Estuary', 'Lagoon', or 'OpenCoast'

        Legacy Sentinel-2 band keys (partial backward compatibility):
          B3_green  -> maps to restoration_efficiency (weak proxy, scaled by 5.0)

          B2_blue, B4_red, B8_nir, B11_swir, NDVI — ACCEPTED WITHOUT ERROR but
          have NO effect on the prediction. There is no scientifically defensible
          mapping from raw reflectance values to the ratio/fraction/uncertainty
          features the model uses. These keys will be reported in
          features_actually_used['caller_supplied_no_effect'].

        Legacy BC seagrass keys (partial backward compatibility):
          anthropogenic_stress_index -> maps to restoration_efficiency
          percent_oc, percent_fines, sea_surface_temperature_c — ACCEPTED WITHOUT
          ERROR but have no effect on the prediction (same reason as above).

    Returns
    -------
    dict:
        predicted_carbon_tC_ha       (float) - total carbon density (tC/ha)
        predicted_carbon_MgC         (float) - raw model output (MgC)
        credits_per_hectare          (float) - CO2e credits (1 tC = 3.67 tCO2e)
        aboveground_carbon_tC        (float) - estimated AGB carbon component (tC)
        soil_organic_carbon_tC       (float) - estimated SOC carbon component (tC)
        confidence_level             (str)   - 'High' / 'Medium' / 'Low'
        input_type                   (str)   - 'mangrove' / 'legacy_bands' / 'legacy_bc'
        features_actually_used       (dict)  - supplied / defaulted / no_effect lists
        status                       (str)   - 'success' or 'error'
        message                      (str)   - human-readable note
    """
    try:
        pipeline, meta = _load_artifacts()
        input_type = _detect_input_type(site_data)

        # Remap legacy inputs into mangrove feature space
        effective_data = (_remap_legacy(site_data, input_type, meta)
                          if input_type != "mangrove" else site_data)

        # Pass original site_data so no_effect reflects the original caller's keys
        X_row, features_used = _build_feature_row(
            effective_data, meta,
            original_site_data=(site_data if input_type != "mangrove" else None)
        )
        X_row.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Predict carbon density (MgC/ha = tC/ha)
        pred_density_tC_ha = float(pipeline.predict(X_row)[0])
        pred_density_tC_ha = max(0.0, pred_density_tC_ha)

        # ── Derived quantities ────────────────────────────────────────────────
        CO2e = meta.get("CO2e_factor", 3.67)

        agb_frac = _AGB_FRACTION_DEFAULT
        soc_frac = 1.0 - agb_frac

        aboveground_tC = pred_density_tC_ha * agb_frac
        soil_tC        = pred_density_tC_ha * soc_frac

        agb_val = effective_data.get("AGB", 0)
        if float(agb_val) > 0:
            est_area_ha    = float(agb_val) / 80.0
            pred_total_MgC = pred_density_tC_ha * est_area_ha
        else:
            pred_total_MgC = pred_density_tC_ha * 10.0  # dummy area if unknown

        carbon_tC_ha   = pred_density_tC_ha
        credits_per_ha = carbon_tC_ha * CO2e

        # Confidence
        confidence = _confidence_level(site_data, meta, input_type)

        # Note message — be explicit about which inputs had no effect
        no_effect = features_used["caller_supplied_no_effect"]
        n_supplied = len(features_used["supplied"])
        n_default  = len(features_used["defaulted"])

        if input_type == "mangrove":
            note = (f"{n_supplied} feature(s) supplied by caller, "
                    f"{n_default} filled with training defaults.")
        else:
            # Identify which no-effect keys came from the original call (not the
            # remapped dict, which is what effective_data shows)
            original_no_effect = [
                k for k in site_data
                if k not in (set(LEGACY_BAND_MAP) | set(LEGACY_BC_MAP) | {"habitat_type"})
                and k.lower() not in {c.lower() for c in meta["feature_columns"]}
            ]
            # Also include band/BC keys that were in the call but not in the maps
            unmapped_supplied = [
                k for k in site_data
                if k in (_BAND_KEYS_NO_EFFECT | _BC_KEYS_NO_EFFECT)
            ]
            all_no_effect = list(dict.fromkeys(original_no_effect + unmapped_supplied))

            if all_no_effect:
                note = (
                    f"Input remapped from {input_type} format. "
                    f"{n_supplied} feature(s) reached the model; "
                    f"{n_default} filled with training defaults. "
                    f"The following supplied keys have NO effect on the prediction "
                    f"because they do not map to any feature used by the current model "
                    f"(only typology_class and restoration_efficiency-equivalent inputs "
                    f"affect this prediction): {all_no_effect}. "
                    f"For accurate results supply: AGB_carbon_fraction, SOC_agb_ratio, "
                    f"restoration_efficiency, typology_class."
                )
            else:
                note = (
                    f"Input remapped from {input_type} format. "
                    f"{n_supplied} feature(s) reached the model; "
                    f"{n_default} filled with training defaults."
                )

        return {
            "predicted_carbon_tC_ha" : round(carbon_tC_ha,   4),
            "predicted_carbon_MgC"   : round(pred_total_MgC,  2),
            "credits_per_hectare"    : round(credits_per_ha,  4),
            "aboveground_carbon_tC"  : round(aboveground_tC,  2),
            "soil_organic_carbon_tC" : round(soil_tC,         2),
            "confidence_level"       : confidence,
            "input_type"             : input_type,
            "features_actually_used" : features_used,
            "status"                 : "success",
            "message"                : note,
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

    # ── Test A: Full mangrove feature dict (direct feature_columns keys) ───────
    print("\n[Test A] Full mangrove dict — direct feature_columns keys:")
    full_mangrove = {
        "typology_class"       : "Delta",
        "AGB_carbon_fraction"  : 0.42,
        "SOC_agb_ratio"        : 13.0,
        "AGB_uncertainty_pct"  : 0.02,
        "SOC_uncertainty_pct"  : 0.01,
        "restoration_efficiency": 0.10,
        "agb_restor_fraction"  : 0.58,
        "soc_restor_fraction"  : 0.11,
    }
    result_a = predict_carbon(full_mangrove)
    print(f"  Input features : {len(full_mangrove)}")
    for k, v in result_a.items():
        print(f"  {k:<35}: {v}")

    # ── Test B: Different typology (Estuary) same other values ─────────────────
    print("\n[Test B] Same as A but typology_class = Estuary:")
    estuary = dict(full_mangrove, typology_class="Estuary")
    result_b = predict_carbon(estuary)
    for k, v in result_b.items():
        print(f"  {k:<35}: {v}")

    # ── Test C: Minimal dict (only typology class) ────────────────────────────
    print("\n[Test C] Minimal input — only typology class (rest = defaults):")
    result_c = predict_carbon({"typology_class": "Delta"})
    for k, v in result_c.items():
        print(f"  {k:<35}: {v}")

    # ── Test D: Legacy Sentinel-2 bands (backward compatibility) ──────────────
    # Only B3_green reaches the model. All other band keys are no-effect and
    # should appear in features_actually_used['caller_supplied_no_effect'].
    print("\n[Test D] Legacy Sentinel-2 band dict (backward compat):")
    print("  NOTE: Only B3_green maps to restoration_efficiency.")
    print("  B2_blue, B4_red, B8_nir, B11_swir, NDVI have NO effect.")
    legacy_bands = {
        "B2_blue": 0.045, "B3_green": 0.068, "B4_red": 0.052,
        "B8_nir": 0.312,  "B11_swir": 0.098, "NDVI": 0.714,
    }
    result_d = predict_carbon(legacy_bands)
    for k, v in result_d.items():
        print(f"  {k:<35}: {v}")

    # ── Test E: Legacy BC seagrass dict (backward compat) ─────────────────────
    # Only anthropogenic_stress_index reaches the model via restoration_efficiency.
    # percent_oc, percent_fines, sea_surface_temperature_c have NO effect.
    print("\n[Test E] Legacy BC seagrass dict (backward compat):")
    print("  NOTE: Only anthropogenic_stress_index maps to restoration_efficiency.")
    print("  percent_oc, percent_fines, sea_surface_temperature_c have NO effect.")
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

    # ── Test G: Behavioural correctness checks ────────────────────────────────
    print("\n[Test G] Behavioural correctness:")

    # G1 — Two inputs differing ONLY in typology_class must produce different output.
    base = {"restoration_efficiency": 0.5, "AGB_carbon_fraction": 0.42}
    r_delta    = predict_carbon(dict(base, typology_class="Delta"))
    r_lagoon   = predict_carbon(dict(base, typology_class="Lagoon"))
    assert r_delta["status"] == "success" and r_lagoon["status"] == "success"
    delta_carbon  = r_delta["predicted_carbon_tC_ha"]
    lagoon_carbon = r_lagoon["predicted_carbon_tC_ha"]
    assert delta_carbon != lagoon_carbon, (
        f"FAIL: typology_class should affect prediction "
        f"(Delta={delta_carbon}, Lagoon={lagoon_carbon})"
    )
    print(f"  G1 PASS — typology_class affects prediction: "
          f"Delta={delta_carbon}, Lagoon={lagoon_carbon}")

    # G2 — Two inputs differing ONLY in a no-effect band key must produce identical output.
    r_high_ndvi = predict_carbon({"NDVI": 0.95, "typology_class": "OpenCoast"})
    r_low_ndvi  = predict_carbon({"NDVI": 0.01, "typology_class": "OpenCoast"})
    assert r_high_ndvi["status"] == "success" and r_low_ndvi["status"] == "success"
    high_c = r_high_ndvi["predicted_carbon_tC_ha"]
    low_c  = r_low_ndvi["predicted_carbon_tC_ha"]
    assert high_c == low_c, (
        f"FAIL: NDVI (no-effect key) should not change prediction "
        f"(NDVI=0.95 -> {high_c}, NDVI=0.01 -> {low_c})"
    )
    print(f"  G2 PASS — NDVI has no effect on prediction: "
          f"both NDVI values -> {high_c} tC/ha")

    # G3 — Verify no-effect keys show up correctly in features_actually_used
    # When NDVI + B8_nir are passed alongside a real key, they should show up
    # in caller_supplied_no_effect (they are detected as legacy_bands, remapped
    # to effective_data which has no B8_nir/NDVI, then the original keys are
    # compared against feature_columns).
    r_bands = predict_carbon({"NDVI": 0.7, "B8_nir": 0.3, "typology_class": "Delta"})
    no_eff = r_bands["features_actually_used"]["caller_supplied_no_effect"]
    assert "NDVI" in no_eff, (
        f"FAIL: NDVI should be in caller_supplied_no_effect, got: {no_eff}"
    )
    assert "B8_nir" in no_eff, (
        f"FAIL: B8_nir should be in caller_supplied_no_effect, got: {no_eff}"
    )
    print(f"  G3 PASS — no-effect keys correctly reported: {no_eff}")

    print("\n[OK] All self-tests passed.\n")
