import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from backend.app import app
import src.live_satellite as ls
import json

# Mock the STAC API call to prevent network access issues
def mock_fetch(*args, **kwargs):
    return {
        "status": "success",
        "metadata": {
            "scene_id": "MOCK_SCENE",
            "date_acquired": "2023-01-01T00:00:00Z",
            "cloud_cover_percent": 5.0
        },
        "bands": {
            "B2_blue": 0.04,
            "B3_green": 0.05,
            "B4_red": 0.03,
            "B8_nir": 0.35,
            "B11_swir": 0.1,
            "NDVI": 0.6
        }
    }
ls.fetch_live_sentinel2_bands = mock_fetch

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
