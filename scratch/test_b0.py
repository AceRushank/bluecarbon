import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

print("1. Running /api/estimate to setup pending validation...")
est_resp = client.post("/api/estimate", json={
    "site_id": "TEST-B0-001",
    "latitude": 21.9,
    "longitude": 88.5,
    "area_hectares": 500.0,
    "typology": "Delta",
    "cloud_cover": 5.0
})
if est_resp.status_code != 200:
    print(f"Estimate failed: {est_resp.text}")
    sys.exit(1)
site_id = est_resp.json()["site_id"]
print(f"Got site_id: {site_id}")

print("2. Running /api/verify-and-mint...")
mint_resp = client.post("/api/verify-and-mint", 
    headers={"X-Api-Key": "bluecarbon_oracle_2025_secure"},
    json={
        "site_id": site_id,
        "user_address": "0xb7f611111AC0228799bFBbF5BEbf1E6B6ddD4e83",
        "latitude": "21.9",
        "longitude": "88.5",
        "area_hectares": 500.0,
        "owner_address": "0xb7f611111AC0228799bFBbF5BEbf1E6B6ddD4e83"
    }
)
if mint_resp.status_code != 200:
    print(f"Mint failed: {mint_resp.text}")
    sys.exit(1)
print("Mint successful!")

print("3. Querying /verify/{site_id}...")
resp = client.get(f"/verify/{site_id}")
print(f"Status Code: {resp.status_code}")
print(f"HTML Content Snippet: {resp.text[:500]}...")
if "Project Not Found" in resp.text:
    print("WARNING: Rendered error page!")
else:
    print("SUCCESS: Rendered valid template!")
