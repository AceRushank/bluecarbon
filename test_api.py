import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from backend.app import app
import json

client = TestClient(app)

print("--- Testing Delta ---")
res1 = client.post("/api/estimate", json={"site_id": "T1", "latitude": 20.7, "longitude": 86.8, "area_hectares": 100, "typology_class": "Delta"}).json()
print("Density tC/ha:", res1["carbon_density_tC_ha"])
print("Total tC:", res1["total_carbon_stock_tC"])
print("Provenance:", json.dumps(res1.get("data_provenance", {})))

print("\n--- Testing OpenCoast ---")
res2 = client.post("/api/estimate", json={"site_id": "T2", "latitude": 20.7, "longitude": 86.8, "area_hectares": 100, "typology_class": "OpenCoast"}).json()
print("Density tC/ha:", res2["carbon_density_tC_ha"])
print("Total tC:", res2["total_carbon_stock_tC"])

print("\n--- Testing Area 200 (Delta) ---")
res3 = client.post("/api/estimate", json={"site_id": "T3", "latitude": 20.7, "longitude": 86.8, "area_hectares": 200, "typology_class": "Delta"}).json()
print("Density tC/ha:", res3["carbon_density_tC_ha"])
print("Total tC:", res3["total_carbon_stock_tC"])
