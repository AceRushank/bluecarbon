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

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Blockchain dependencies
cd blockchain && npm install

# 4. Start local Hardhat node (in one terminal)
npx hardhat node

# 5. Deploy contracts (in a second terminal)
npx hardhat run scripts/deploy.js --network hardhat
# Note: This generates blockchain/deployed_contracts.json, which is gitignored and must be regenerated locally.

# 6. Run the FastAPI backend
cd .. && uvicorn backend.app:app --reload --port 8000
```

> **Note on backend/app.py Fallback Logic:** The backend contains fallback address logic and mock data responses. This exists so that `app.py` doesn't hard-fail if the smart contract deployment hasn't been run yet (e.g., if `deployed_contracts.json` is missing or the node isn't up).

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
pandas · numpy · scikit-learn · matplotlib · joblib · pydantic · fastapi · uvicorn · web3 · python-dotenv · pystac-client
```
