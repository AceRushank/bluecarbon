"""
src/credit_scorer.py
---------------------
Carbon Credit Quality Score calculator.
Produces a composite 0-100 score and letter grade (BB–AAA)
from observable site metrics, satellite data quality, and data provenance.
"""


def calculate_credit_score(
    ndvi: float,
    carbon_density: float,
    cloud_cover: float,
    gmw_validated: bool,
    is_ground_truth: bool = True,
    restoration_fraction: float = 0.5,
    model_confidence: float = 0.8,
    typology_mean: float = 173.34,
) -> dict:
    """
    Calculates 0-100 composite credit quality score with explicit data provenance weighting:
    - Data Provenance      : 0–15 pts (15 for field plots, 3 for synthetic/formula data)
    - GMW Location Valid   : 0–20 pts
    - NDVI Health          : 0–20 pts
    - Carbon Density       : 0–15 pts
    - Restoration Fraction : 0–15 pts
    - Scene Quality        : 0–10 pts
    - Model Confidence     : 0–5  pts
    Total                  : 100 pts
    """
    scores = {}

    # 1. Data Provenance (0–15 points)
    # Field-measured ground-truth data (Sundarbans 76 plots) receives full 15 pts.
    # Formula-derived / synthetic targets (Andaman) receive 3 pts to reflect lack of field calibration.
    scores["data_provenance"] = 15.0 if is_ground_truth else 3.0

    # 2. GMW boundary validation (0–20 points)
    scores["location_verified"] = 20.0 if gmw_validated else 0.0

    # 3. NDVI vegetation health (0–20 points)
    scores["ndvi_health"] = round(min(ndvi / 0.9, 1.0) * 20, 1)

    # 4. Carbon density vs typology mean (0–15 points)
    ratio = min(carbon_density / typology_mean, 1.0)
    scores["carbon_density_score"] = round(ratio * 15, 1)

    # 5. Restoration potential (0–15 points)
    scores["restoration_potential"] = round(
        min(max(restoration_fraction, 0), 1) * 15, 1
    )

    # 6. Scene quality — cloud cover penalty (0–10 points)
    scores["scene_quality"] = round(max(0, (1 - cloud_cover / 100)) * 10, 1)

    # 7. Model confidence (0–5 points)
    scores["model_confidence"] = round(
        min(max(model_confidence, 0), 1) * 5, 1
    )

    total = round(sum(scores.values()), 1)

    if total >= 88:
        grade  = "AAA"
        color  = "#16a34a"
        market = "Premium tier — $45–50 / credit"
    elif total >= 78:
        grade  = "AA"
        color  = "#22c55e"
        market = "High quality — $35–45 / credit"
    elif total >= 68:
        grade  = "A"
        color  = "#84cc16"
        market = "Standard tier — $25–35 / credit"
    elif total >= 58:
        grade  = "BBB"
        color  = "#f97316"
        market = "Below standard — $15–25 / credit"
    else:
        grade  = "BB"
        color  = "#ef4444"
        market = "Requires review — $10–15 / credit"

    return {
        "total_score": total,
        "grade":       grade,
        "color":       color,
        "market_tier": market,
        "breakdown":   scores,
    }
