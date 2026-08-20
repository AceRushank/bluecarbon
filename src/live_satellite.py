"""
src/live_satellite.py
---------------------
Queries the open Earth Search AWS STAC catalog for recent, cloud-free 
Sentinel-2 L2A imagery over a given coordinate. Extracts metadata and 
simulates radiometric band values indicative of healthy mangrove cover
for the prototype pipeline.
"""

from pystac_client import Client
import datetime
import random

# Earth Search by Element 84 hosts Sentinel-2 COGs on AWS
STAC_API_URL = "https://earth-search.aws.element84.com/v1"

def fetch_live_sentinel2_bands(lat: float, lon: float):
    """
    Search STAC for recent Sentinel-2 imagery for the given coordinates.
    Returns simulated spectral band data and real scene metadata.
    """
    # Create a small bounding box (approx ~1km around point)
    delta = 0.01 
    bbox = [lon - delta, lat - delta, lon + delta, lat + delta]
    
    # Search for imagery from the last 6 months
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=180)
    time_range = f"{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    try:
        catalog = Client.open(STAC_API_URL)
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=time_range,
            query={"eo:cloud_cover": {"lt": 15}},  # less than 15% clouds
            max_items=1
        )
        
        items = list(search.items())
        
        if not items:
            return {
                "status": "error",
                "message": f"No cloud-free imagery found for coordinates ({lat}, {lon}) in the last 6 months."
            }
            
        item = items[0]
        cloud_cover = item.properties.get("eo:cloud_cover", 0.0)
        date_acquired = item.properties.get("datetime")
        scene_id = item.id
        
        # PROTOTYPE SIMULATION:
        # Instead of fully extracting COG pixels via rasterio (heavy/slow),
        # we simulate typical reflectance values for dense coastal mangroves.
        # Mangroves usually have high NIR (B8), low Red (B4), and moderate SWIR (B11).
        
        # Base healthy mangrove signature
        b2_blue = random.uniform(0.02, 0.05)
        b3_green = random.uniform(0.04, 0.08)
        b4_red = random.uniform(0.02, 0.06)
        b8_nir = random.uniform(0.25, 0.45) 
        b11_swir = random.uniform(0.08, 0.15)
        
        # Calculate dynamic NDVI
        ndvi = (b8_nir - b4_red) / (b8_nir + b4_red)
        
        return {
            "status": "success",
            "metadata": {
                "scene_id": scene_id,
                "date_acquired": date_acquired,
                "cloud_cover_percent": round(cloud_cover, 2)
            },
            "bands": {
                "B2_blue": round(b2_blue, 4),
                "B3_green": round(b3_green, 4),
                "B4_red": round(b4_red, 4),
                "B8_nir": round(b8_nir, 4),
                "B11_swir": round(b11_swir, 4),
                "NDVI": round(ndvi, 4)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"STAC API query failed: {str(e)}"
        }

# Quick test if run directly
if __name__ == "__main__":
    # Test on Bhitarkanika coordinates
    res = fetch_live_sentinel2_bands(20.7211, 86.8880)
    print(res)
