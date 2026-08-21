import os
import json
import hashlib
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Header
import re
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from web3 import Web3
from web3.exceptions import ContractLogicError
from dotenv import load_dotenv

import joblib
import numpy as np
import time

# Ensure we can import src modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.live_satellite import fetch_live_sentinel2_bands
from src.gmw_validator import get_validator
from src.credit_scorer import calculate_credit_score

# ── Load Sundarbans RandomForest model ────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
try:
    _RF_MODEL = joblib.load(_MODEL_PATH)
    print(f"[MODEL]  Loaded model.pkl from {_MODEL_PATH}")
except FileNotFoundError:
    _RF_MODEL = None
    print(f"[MODEL]  WARNING: model.pkl not found at {_MODEL_PATH}. Run src/train_model.py first.")

def predict_carbon_density(ndvi: float, lat: float, lon: float) -> float:
    """Predict carbon density (tC/ha) using [NDVI, lat, lon] features."""
    if _RF_MODEL is None:
        raise RuntimeError("model.pkl not loaded — run src/train_model.py")
    X = np.array([[ndvi, lat, lon]])
    return float(_RF_MODEL.predict(X)[0])

# ── 1. Init & Config ──────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), '../blockchain/.env'))

NETWORK_CONFIG = {
    "sepolia": {"chain_id": 11155111, "rpc_env": "SEPOLIA_RPC_URL", "fallback": "https://rpc.sepolia.org"},
    "amoy": {"chain_id": 80002, "rpc_env": "AMOY_RPC_URL", "fallback": "https://rpc-amoy.polygon.technology/"},
    "local": {"chain_id": 31337, "rpc_env": "RPC_URL", "fallback": "http://127.0.0.1:8545"}
}

ACTIVE_NETWORK = os.getenv("NETWORK", "local").lower()
if ACTIVE_NETWORK not in NETWORK_CONFIG:
    ACTIVE_NETWORK = "local"

cfg = NETWORK_CONFIG[ACTIVE_NETWORK]
CHAIN_ID = cfg["chain_id"]
RPC_URL = os.getenv(cfg["rpc_env"], cfg["fallback"])

# Fallback to Hardhat Account #0 if no key is provided
ORACLE_PRIVATE_KEY = os.getenv("PRIVATE_KEY") or os.getenv("ORACLE_PRIVATE_KEY") or "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
MINT_API_KEY = os.getenv("MINT_API_KEY", "bluecarbon_oracle_2025_secure")

# Initialize Web3
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# Fallback values
REGISTRY_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
TOKEN_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"

# Load Contract Data
contracts_file = os.path.join(os.path.dirname(__file__), '../blockchain/deployed_contracts.json')
try:
    with open(contracts_file, 'r', encoding='utf-8') as f:
        deployed_data_all = json.load(f)
        
    deployed_data = deployed_data_all.get(ACTIVE_NETWORK, {})
    if not deployed_data:
        print(f"Warning: No deployment found for network '{ACTIVE_NETWORK}'.")
        
    registry_data = deployed_data.get("contracts", {}).get("BlueCarbonRegistry", {})
    token_data = deployed_data.get("contracts", {}).get("CarbonCreditToken", {})
    
    REGISTRY_ADDRESS = registry_data.get("address", REGISTRY_ADDRESS)
    REGISTRY_ABI = registry_data.get("abi")
    TOKEN_ADDRESS = token_data.get("address", TOKEN_ADDRESS)
    TOKEN_ABI = token_data.get("abi")
    
    if REGISTRY_ABI and TOKEN_ABI:
        registry_contract = w3.eth.contract(address=REGISTRY_ADDRESS, abi=REGISTRY_ABI)
        token_contract = w3.eth.contract(address=TOKEN_ADDRESS, abi=TOKEN_ABI)
    else:
        print("Warning: Missing ABIs in deployed_contracts.json")
        registry_contract = None
        token_contract = None
except Exception as e:
    print(f"Warning: Could not load contract ABIs: {e}")
    registry_contract = None
    token_contract = None

print("="*60)
print(f"Blue Carbon MRV API Started")
print(f"Network    : {ACTIVE_NETWORK} (Chain ID: {CHAIN_ID})")
from urllib.parse import urlparse
try:
    parsed_url = urlparse(RPC_URL)
    masked_rpc = f"{parsed_url.scheme}://{parsed_url.netloc}/...[MASKED]"
except:
    masked_rpc = "...[MASKED]"

print(f"RPC URL    : {masked_rpc}")
print(f"Registry   : {REGISTRY_ADDRESS}")
print(f"Token      : {TOKEN_ADDRESS}")
print("="*60)

import time
MOCK_REGISTRY = []
pending_validations = {}

# Initialize FastAPI app
app = FastAPI(title="Blue Carbon MRV API")

# Mount templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), '../frontend'))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ───────────────────────────────────────────────────────────
class EstimateRequest(BaseModel):
    site_id: str = Field(..., min_length=5, max_length=30, pattern=r"^[A-Z0-9\-]+$")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    area_hectares: float = Field(..., gt=0, le=50000, description="Max 50,000 ha per project")
    manual_ndvi: Optional[float] = None

class VerifyRequest(BaseModel):
    site_id: str
    latitude: str
    longitude: str
    area_hectares: float
    owner_address: str

class RetireRequest(BaseModel):
    amount: float
    reason: str
    user_private_key: Optional[str] = None

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "node_connected": w3.is_connected(),
        "chain_id": CHAIN_ID,
        "contracts": {
            "registry": REGISTRY_ADDRESS,
            "token": TOKEN_ADDRESS if 'TOKEN_ADDRESS' in globals() else None
        }
    }

@app.post("/api/estimate")
def estimate_carbon(req: EstimateRequest):
    # ── Sundarbans bounding box check ───────────────────────────────────────
    # Model trained on lat 21.2–23.1, lon 87.8–89.9 only
    SUND_LAT_MIN, SUND_LAT_MAX = 21.2, 23.1
    SUND_LON_MIN, SUND_LON_MAX = 87.8, 89.9
    if not (SUND_LAT_MIN <= req.latitude <= SUND_LAT_MAX and
            SUND_LON_MIN <= req.longitude <= SUND_LON_MAX):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "OUT_OF_REGION",
                "message": "Coordinates outside Sundarbans model training region "
                           f"(lat {SUND_LAT_MIN}–{SUND_LAT_MAX}, "
                           f"lon {SUND_LON_MIN}–{SUND_LON_MAX})."
            }
        )

    # ── GMW spatial validation ──────────────────────────────────────────────
    validator = get_validator()
    gmw_result = validator.validate(req.latitude, req.longitude)

    # ── Fetch live Sentinel-2 scene (gets real scene_id, cloud cover, NDVI) ─
    stac_res = fetch_live_sentinel2_bands(req.latitude, req.longitude)
    if stac_res["status"] == "error":
        raise HTTPException(status_code=400, detail=stac_res["message"])

    cloud_cover = float(stac_res["metadata"].get("cloud_cover_percent", 0))
    if cloud_cover > 20:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "HIGH_CLOUD_COVER",
                "message": f"Cloud cover is {cloud_cover:.1f}% — exceeds 20% threshold. Sentinel-2 revisit: 5 days.",
                "cloud_cover": cloud_cover,
                "retry_in_days": 5
            }
        )

    # ── Determine NDVI to use ───────────────────────────────────────────────
    bands = stac_res["bands"]
    ndvi_val = req.manual_ndvi if req.manual_ndvi is not None else float(bands.get("NDVI", 0.65))

    # ── Run Sundarbans RandomForest model: [NDVI, lat, lon] → tC/ha ────────
    try:
        tC_ha = predict_carbon_density(ndvi_val, req.latitude, req.longitude)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── AGB / SOC split (28% / 72%) ────────────────────────────────────────
    # Based on Murdiyarso et al. mangrove carbon partitioning constants
    agb_density  = tC_ha * 0.28   # Above-Ground Biomass (tC/ha)
    soc_density  = tC_ha * 0.72   # Soil Organic Carbon  (tC/ha)

    # Scale up to the full site area
    total_tC = tC_ha       * req.area_hectares
    agb_tC   = agb_density * req.area_hectares
    soc_tC   = soc_density * req.area_hectares

    # ── Convert to CO2 equivalents: 1 tC = 3.67 tCO2e ──────────────────────
    total_credits = total_tC * 3.67   # $BCO2 tokens to mint

    # ── Credit quality score ────────────────────────────────────────────────
    credit_score = calculate_credit_score(
        ndvi=ndvi_val,
        carbon_density=tC_ha,
        cloud_cover=cloud_cover,
        gmw_validated=gmw_result["gmw_validated"],
        restoration_fraction=0.5,
        model_confidence=0.8,
        typology_mean=173.34,   # Sundarbans dataset mean
    )

    # ── Cache for server-side use in /api/verify-and-mint ──────────────────
    pending_validations[req.site_id] = {
        "gmw_validated": gmw_result["gmw_validated"],
        "fraud_flag":    gmw_result["fraud_flag"],
        "gmw_zone":      gmw_result["gmw_zone"],
        "lat":           req.latitude,
        "lon":           req.longitude,
        "carbon_tons":   int(total_credits),
        "credit_score":  credit_score["total_score"],
        "credit_grade":  credit_score["grade"],
        "timestamp":     time.time()
    }

    return {
        "site_id":                 req.site_id,
        "NDVI":                    ndvi_val,
        "satellite_meta": {
            "scene_id":            stac_res["metadata"].get("scene_id", "N/A"),
            "cloud_cover_percent": cloud_cover,
            "NDVI":                ndvi_val
        },
        "carbon_density_tC_ha":    tC_ha,
        "agb_density_tC_ha":       agb_density,
        "soc_density_tC_ha":       soc_density,
        "aboveground_biomass_tC":  agb_tC,
        "soil_organic_carbon_tC": soc_tC,
        "total_carbon_stock_tC":   total_tC,
        "total_credits_tCO2e":     total_credits,
        "predicted_credits":       total_credits,
        "gmw_validated":           gmw_result["gmw_validated"],
        "gmw_zone":                gmw_result["gmw_zone"],
        "fraud_flag":              gmw_result["fraud_flag"],
        "gmw_warning":             gmw_result["warning"],
        "credit_score":            credit_score,
        "model_info": {
            "model":    "RandomForestRegressor",
            "features": ["NDVI", "lat", "lon"],
            "dataset":  "Sundarbans Ground-Truth (76 plots, 2023)",
            "agb_fraction": 0.28,
            "soc_fraction": 0.72,
            "co2e_factor":  3.67
        }
    }

@app.post("/api/verify-and-mint")
def verify_and_mint(req: VerifyRequest, x_api_key: str = Header(None)):
    if x_api_key != MINT_API_KEY:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "UNAUTHORIZED",
                "message": "Valid API key required for minting operations."
            }
        )

    if not re.match(r'^[A-Z0-9\-]{5,30}$', req.site_id):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_SITE_ID",
                "message": "Site ID must be 5-30 uppercase alphanumeric characters and hyphens only."
            }
        )
        
    stored_time = pending_validations.get(req.site_id, {}).get("timestamp", 0)
    if time.time() - stored_time > 3600:
        if req.site_id in pending_validations:
            del pending_validations[req.site_id]
        raise HTTPException(
            status_code=400,
            detail={
                "error": "VALIDATION_EXPIRED",
                "message": "Estimate has expired. Please re-run the satellite estimate before minting."
            }
        )

    carbon_tons = int(pending_validations[req.site_id]["carbon_tons"])

    oracle_account = w3.eth.account.from_key(ORACLE_PRIVATE_KEY)
    
    # Generate mock IPFS Proof Hash
    payload = req.model_dump_json()
    hash_hex = hashlib.sha256(payload.encode()).hexdigest()
    ipfs_hash = f"ipfs://mock_{hash_hex[:16]}"
    
    if registry_contract:
        try:
            tx = registry_contract.functions.verifyAndIssueCredits(
                req.site_id,
                carbon_tons,
                ipfs_hash
            ).build_transaction({
                'from': oracle_account.address,
                'nonce': w3.eth.get_transaction_count(oracle_account.address),
                'gasPrice': w3.eth.gas_price
            })
            
            signed_tx = w3.eth.account.sign_transaction(tx, private_key=ORACLE_PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "status": "success",
                "tx_hash": tx_hash.hex(),
                "block_number": receipt.blockNumber,
                "gas_used": receipt.gasUsed,
                "ipfs_proof": ipfs_hash
            }
        except Exception as e:
            print(f"Web3 transaction failed: {e}. Falling back to mock data.")
            
    # Fallback logic
    existing = next((p for p in MOCK_REGISTRY if p["site_id"] == req.site_id), None)
    if existing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ALREADY_MINTED",
                "message": f"Site {req.site_id} has already been registered in the registry."
            }
        )
        
    mock_tx_hash = "0x" + hashlib.sha256((payload + "tx").encode()).hexdigest()
    MOCK_REGISTRY.append({
        "site_id": req.site_id,
        "owner": req.owner_address,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "area_hectares": req.area_hectares,
        "is_verified": True,
        "carbon_tons": carbon_tons,
        "credits_minted": carbon_tons, # assuming 1 to 1 for prototype mock
        "ipfs_proof_hash": ipfs_hash,
        "timestamp": int(time.time())
    })
    
    return {
        "status": "success (mock fallback)",
        "tx_hash": mock_tx_hash,
        "block_number": 999999,
        "gas_used": 21000,
        "ipfs_proof": ipfs_hash
    }

@app.get("/api/registry/projects")
def get_registry_projects():
    projects = []
    if registry_contract:
        try:
            site_ids = registry_contract.functions.getAllSiteIds().call()
            for sid in site_ids:
                p = registry_contract.functions.getProject(sid).call()
                projects.append({
                    "site_id": p[0],
                    "owner": p[1],
                    "latitude": p[2],
                    "longitude": p[3],
                    "area_hectares": p[4],
                    "is_verified": p[5],
                    "carbon_tons": p[6],
                    "credits_minted": str(w3.from_wei(p[7], 'ether')),
                    "ipfs_proof_hash": p[8],
                    "timestamp": p[9]
                })
        except Exception as e:
            print(f"Web3 registry fetch failed: {e}. Falling back to mock data.")
            
    projects.extend(MOCK_REGISTRY)
    return {"projects": projects}

@app.get("/verify/{project_id}")
async def verify_page(request: Request, project_id: str):
    if not registry_contract:
        return templates.TemplateResponse(request=request, name="verify.html", context={
            "error": "Registry contract not initialized (mock mode).",
            "network_name": ACTIVE_NETWORK
        })
    
    try:
        project = registry_contract.functions.getProject(project_id).call()
        print("="*40)
        print(f"Raw Project Tuple for {project_id}:")
        print(project)
        print("="*40)
        
        if not project[0]:
            return templates.TemplateResponse(request=request, name="verify.html", context={
                "error": "Project Not Found",
                "network_name": ACTIVE_NETWORK
            })
            
        return templates.TemplateResponse(request=request, name="verify.html", context={
            "project_id": project_id,
            "raw_data": project,
            "error": None,
            "network_name": ACTIVE_NETWORK
        })
    except Exception as e:
        print(f"Error in verify_page: {e}")
        return templates.TemplateResponse(request=request, name="verify.html", context={
            "error": "Project Not Found",
            "network_name": ACTIVE_NETWORK
        })

@app.post("/api/retire")
def retire_credits(req: RetireRequest):
    if not registry_contract:
        raise HTTPException(status_code=500, detail="Contract not configured")
        
    pk = req.user_private_key or ORACLE_PRIVATE_KEY
    if not pk:
        raise HTTPException(status_code=400, detail="Private key required to retire credits")
        
    user_account = w3.eth.account.from_key(pk)
    amount_wei = w3.to_wei(req.amount, 'ether')
    
    try:
        # Step 1: Approve the registry to spend BCO2
        approve_tx = token_contract.functions.approve(
            REGISTRY_ADDRESS,
            amount_wei
        ).build_transaction({
            'from': user_account.address,
            'nonce': w3.eth.get_transaction_count(user_account.address),
            'gasPrice': w3.eth.gas_price
        })
        signed_approve = w3.eth.account.sign_transaction(approve_tx, private_key=pk)
        w3.eth.send_raw_transaction(signed_approve.rawTransaction)
        # Wait slightly or just use the next nonce, wait for receipt is safer
        w3.eth.wait_for_transaction_receipt(signed_approve.hash)
        
        # Step 2: Retire
        retire_tx = registry_contract.functions.retireCredits(
            amount_wei,
            req.reason
        ).build_transaction({
            'from': user_account.address,
            'nonce': w3.eth.get_transaction_count(user_account.address),
            'gasPrice': w3.eth.gas_price
        })
        signed_retire = w3.eth.account.sign_transaction(retire_tx, private_key=pk)
        tx_hash = w3.eth.send_raw_transaction(signed_retire.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "status": "success",
            "tx_hash": tx_hash.hex(),
            "amount_retired": req.amount,
            "reason": req.reason
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Static UI Serving ─────────────────────────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
os.makedirs(frontend_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
