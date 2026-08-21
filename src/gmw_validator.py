"""
src/gmw_validator.py
---------------------
Global Mangrove Watch (GMW 2020) boundary validator.
Checks if a given lat/lon falls within a known mangrove zone,
with a 0.05-degree (~5 km) coastal edge buffer for tolerance.
"""

from shapely.geometry import Point, shape
import json
import os


class GMWValidator:
    def __init__(self, geojson_path="data/gmw_2020.geojson"):
        # Resolve path relative to project root
        if not os.path.isabs(geojson_path):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            geojson_path = os.path.join(root, geojson_path)

        with open(geojson_path) as f:
            data = json.load(f)

        self.features = [
            {
                "name": feat["properties"]["name"],
                "shape": shape(feat["geometry"])
            }
            for feat in data["features"]
        ]

    def validate(self, lat: float, lon: float) -> dict:
        point = Point(lon, lat)
        matched_zone = None

        for feature in self.features:
            # 0.008 degree buffer (~800 m) for coastal edge tolerance
            if (feature["shape"].contains(point) or
                    feature["shape"].distance(point) < 0.008):
                matched_zone = feature["name"]
                break

        is_valid = matched_zone is not None

        return {
            "gmw_validated": is_valid,
            "gmw_zone": matched_zone,
            "fraud_flag": not is_valid,
            "warning": None if is_valid else (
                "Coordinates fall outside recorded mangrove boundaries "
                "(GMW 2020). Site flagged for manual review."
            )
        }


# Singleton — load once at startup
_validator = None


def get_validator():
    global _validator
    if _validator is None:
        _validator = GMWValidator()
    return _validator
