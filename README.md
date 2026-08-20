# 🌿 Blue Carbon MRV — Mangrove Carbon Stock Estimator

A machine-learning pipeline to estimate mangrove carbon stock (tC/ha) from
Sentinel-2 satellite spectral data. Built for the Smart India Hackathon as a
prototype blue-carbon Monitoring, Reporting, and Verification (MRV) system.

**Study site:** Bhitarkanika National Park, Odisha, India

---

## Project Structure

```
bluecarbon/
├── data/
│   └── mangrove_carbon_samples.csv   ← 20 Sentinel-2 + carbon samples
├── models/
│   └── carbon_model.pkl              ← saved Random Forest model
├── outputs/
│   ├── evaluation_report.txt         ← R², RMSE, MAE, feature importance
│   ├── ndvi_vs_carbon.png            ← scatter plot (explore.py)
│   ├── feature_importance.png        ← bar chart (train.py)
│   └── actual_vs_predicted.png       ← test-set predictions (train.py)
├── explore.py      ← Step 1: data loading, stats, NDVI plot
├── train.py        ← Step 2: train RF, evaluate, save model + plots
├── predictor.py    ← Step 3: reusable predict_carbon() + sanity check
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Explore the data
python explore.py

# 4. Train the model (creates models/carbon_model.pkl + outputs/)
python train.py

# 5. Run the predictor sanity check
python predictor.py
```

---

## Using the Predictor in Your Own Code

```python
from predictor import predict_carbon

carbon_tC_ha = predict_carbon({
    "B2_blue":  0.045,
    "B3_green": 0.068,
    "B4_red":   0.052,
    "B8_nir":   0.312,
    "B11_swir": 0.098,
    "NDVI":     0.714,
})
print(f"Estimated carbon stock: {carbon_tC_ha:.2f} tC/ha")
```

---

## Features Used

| Feature   | Sentinel-2 Band | Wavelength   | Role                              |
|-----------|-----------------|--------------|-----------------------------------|
| B2_blue   | Band 2          | 490 nm       | Water / aerosol sensitivity       |
| B3_green  | Band 3          | 560 nm       | Vegetation green peak             |
| B4_red    | Band 4          | 665 nm       | Chlorophyll absorption            |
| B8_nir    | Band 8          | 842 nm       | Biomass / canopy structure        |
| B11_swir  | Band 11         | 1610 nm      | Moisture / soil                   |
| NDVI      | (B8-B4)/(B8+B4) | —            | Vegetation greenness index        |

---

## Model Summary

- **Algorithm:** Random Forest Regressor (200 trees, sklearn)
- **Target:** `carbon_stock_tC_ha` — above-ground carbon in tonnes C/ha
- **Split:** 80% train / 20% test, fixed random seed = 42
- **Validation:** 5-fold cross-validation

See `outputs/evaluation_report.txt` for the full metrics.

---

## Dependencies

```
pandas · numpy · scikit-learn · matplotlib · joblib
```
