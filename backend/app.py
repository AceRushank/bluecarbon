import os
import json
import hashlib
from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from web3 import Web3
from web3.exceptions import ContractLogicError
from dotenv import load_dotenv

# Ensure we can import src modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.live_satellite import fetch_live_sentinel2_bands
from src.predictor import predict_carbon

# ── 1. Init & Config ──────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), '../blockchain/.env'))

RPC_URL = os.getenv("AMOY_RPC_URL", "http://127.0.0.1:8545")
# Fallback to Hardhat Account #0 if no key is provided
ORACLE_PRIVATE_KEY = os.getenv("PRIVATE_KEY") or os.getenv("ORACLE_PRIVATE_KEY") or "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
CHAIN_ID = 80002 if "amoy" in RPC_URL.lower() else 31337

# Initialize Web3
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# Fallback values
REGISTRY_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
TOKEN_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"

# Load Contract Data
contracts_file = os.path.join(os.path.dirname(__file__), '../blockchain/deployed_contracts.json')
try:
    with open(contracts_file, 'r') as f:
        deployed_data = json.load(f)
    registry_data = deployed_data["contracts"]["BlueCarbonRegistry"]
    token_data = deployed_data["contracts"]["CarbonCreditToken"]
    
    REGISTRY_ADDRESS = registry_data.get("address", REGISTRY_ADDRESS)
    REGISTRY_ABI = registry_data["abi"]
    TOKEN_ADDRESS = token_data.get("address", TOKEN_ADDRESS)
    TOKEN_ABI = token_data["abi"]
    
    registry_contract = w3.eth.contract(address=REGISTRY_ADDRESS, abi=REGISTRY_ABI)
    token_contract = w3.eth.contract(address=TOKEN_ADDRESS, abi=TOKEN_ABI)
except Exception as e:
    print(f"Warning: Could not load contract ABIs: {e}")
    registry_contract = None
    token_contract = None

import time
MOCK_REGISTRY = []

# Initialize FastAPI app
app = FastAPI(title="Blue Carbon MRV API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ───────────────────────────────────────────────────────────
class EstimateRequest(BaseModel):
    site_id: str
    latitude: float
    longitude: float
    area_hectares: float
    manual_ndvi: Optional[float] = None

class VerifyRequest(BaseModel):
    site_id: str
    latitude: str
    longitude: str
    area_hectares: float
    predicted_credits: float
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
    # 1. Fetch live satellite imagery (metadata + simulated bands)
    stac_res = fetch_live_sentinel2_bands(req.latitude, req.longitude)
    if stac_res["status"] == "error":
        raise HTTPException(status_code=400, detail=stac_res["message"])
    
    # 2. Build input dictionary for predictor (using STAC bands)
    bands = stac_res["bands"]
    if req.manual_ndvi is not None:
        bands["NDVI"] = req.manual_ndvi
        
    site_data = {
        "typology_class": "OpenCoast", # Default assumption if unknown
        "latitude": req.latitude,
        "longitude": req.longitude,
        **bands
    }
    
    # 3. Call ML Predictor
    pred_res = predict_carbon(site_data)
    if pred_res["status"] == "error":
        raise HTTPException(status_code=500, detail=pred_res["message"])
    
    # 4. Adjust credits based on provided area vs density (tC/ha)
    # Calibrate carbon density using dynamic NDVI scaling
    ndvi_val = req.manual_ndvi if req.manual_ndvi is not None else bands.get("NDVI", 0.0)
    tC_ha = 140.0 + (ndvi_val * 180.0)
    
    # Calculate components
    total_tC = tC_ha * req.area_hectares
    agb_tC = total_tC * 0.28
    soc_tC = total_tC * 0.72
    total_credits = total_tC * 3.67
    
    return {
        "site_id": req.site_id,
        "NDVI": ndvi_val,
        "satellite_meta": {
            "scene_id": stac_res["metadata"].get("scene_id", "N/A"),
            "cloud_cover_percent": stac_res["metadata"].get("cloud_cover_percent", 0.0),
            "NDVI": ndvi_val
        },
        "carbon_density_tC_ha": tC_ha,
        "aboveground_biomass_tC": agb_tC,
        "soil_organic_carbon_tC": soc_tC,
        "total_carbon_stock_tC": total_tC,
        "total_credits_tCO2e": total_credits,
        "predicted_credits": total_credits
    }

@app.post("/api/verify-and-mint")
def verify_and_mint(req: VerifyRequest):
    oracle_account = w3.eth.account.from_key(ORACLE_PRIVATE_KEY)
    
    # Generate mock IPFS Proof Hash
    payload = req.model_dump_json()
    hash_hex = hashlib.sha256(payload.encode()).hexdigest()
    ipfs_hash = f"ipfs://mock_{hash_hex[:16]}"
    carbon_tons = int(req.predicted_credits)
    
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
