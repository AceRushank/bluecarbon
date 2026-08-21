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
import rasterio
from pyproj import Transformer

# Earth Search by Element 84 hosts Sentinel-2 COGs on AWS
STAC_API_URL = "https://earth-search.aws.element84.com/v1"

def fetch_live_sentinel2_bands(lat: float, lon: float):
    """
    Search STAC for recent Sentinel-2 imagery for the given coordinates.
    Returns simulated spectral band data and real scene metadata.
    """
    print(f"[API CALL] fetch_live_sentinel2_bands called with lat={lat}, lon={lon}")
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
            max_items=5
        )
        
        items = list(search.items())
        
        if not items:
            return {
                "status": "error",
                "message": f"No cloud-free imagery found for coordinates ({lat}, {lon}) in the last 6 months."
            }
            
        def read_pixel(url, lat, lon):
            with rasterio.open(url) as src:
                # Reproject lat/lon to match the image's CRS
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                row, col = src.index(x, y)
                # Read a 3x3 pixel window and average
                window = rasterio.windows.Window(col - 1, row - 1, 3, 3)
                data = src.read(1, window=window).astype(float)
                return data.mean() / 10000.0  # Sentinel-2 L2A scale factor

        # Try to find a valid real pixel extraction
        for item in items:
            try:
                print(f"\n[DEBUG] Scene ID: {item.id}")
                print(f"[DEBUG] Available assets: {list(item.assets.keys())}")
                b4_url = item.assets["red"].href
                b8_url = item.assets["nir"].href
                print(f"[DEBUG] b4_url: {b4_url}")
                print(f"[DEBUG] b8_url: {b8_url}")
                
                b4_red = read_pixel(b4_url, lat, lon)
                b8_nir = read_pixel(b8_url, lat, lon)
                
                if b4_red <= 0.0 or b8_nir <= 0.0:
                    print("[SKIP] nodata pixel")
                    continue
                if b4_red > 1.0 or b8_nir > 1.0:
                    print(f"[SKIP] b4 or b8 out of range after scaling: {b4_red}, {b8_nir}")
                    continue
                    
                ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + 1e-10)
                
                return {
                    "status": "success",
                    "metadata": {
                        "scene_id": item.id,
                        "date_acquired": item.properties.get("datetime"),
                        "cloud_cover_percent": round(item.properties.get("eo:cloud_cover", 0.0), 2)
                    },
                    "bands": {
                        "B2_blue": 0.0,
                        "B3_green": 0.0,
                        "B4_red": round(b4_red, 4),
                        "B8_nir": round(b8_nir, 4),
                        "B11_swir": 0.0,
                        "NDVI": round(ndvi, 4)
                    },
                    "ndvi_source": "REAL"
                }
            except Exception as e:
                import traceback
                print(f"[EXCEPTION] {e}")
                print(f"[DEBUG] Rasterio pixel extraction failed for {item.id}: {str(e)}")
                traceback.print_exc()
                continue
                
        print("[FALLBACK] all scenes exhausted or failed")
        # Fallback to modelled if all scenes failed or nodata
        item = items[0] # use metadata from first scene
        b4_red = random.uniform(0.02, 0.06)
        b8_nir = random.uniform(0.25, 0.45) 
        ndvi = (b8_nir - b4_red) / (b8_nir + b4_red)
        
        return {
            "status": "success",
            "metadata": {
                "scene_id": item.id,
                "date_acquired": item.properties.get("datetime"),
                "cloud_cover_percent": round(item.properties.get("eo:cloud_cover", 0.0), 2)
            },
            "bands": {
                "B2_blue": 0.0,
                "B3_green": 0.0,
                "B4_red": round(b4_red, 4),
                "B8_nir": round(b8_nir, 4),
                "B11_swir": 0.0,
                "NDVI": round(ndvi, 4)
            },
            "ndvi_source": "MODELLED"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"STAC API query failed: {str(e)}"
        }

# Quick test if run directly
if __name__ == "__main__":
    # Test on Bhitarkanika coordinates
    res = fetch_live_sentinel2_bands(21.9497, 89.1833)
    print(res)
