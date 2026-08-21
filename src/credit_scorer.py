"""
src/credit_scorer.py
---------------------
Carbon Credit Quality Score calculator.
Produces a composite 0-100 score and letter grade (BB–AAA)
from observable site metrics, used to tier carbon credit
market value estimates.
"""


def calculate_credit_score(
    ndvi: float,
    carbon_density: float,
    cloud_cover: float,
    gmw_validated: bool,
    restoration_fraction: float = 0.5,
    model_confidence: float = 0.8,
    typology_mean: float = 200.0,
) -> dict:
    scores = {}

    # 1. NDVI vegetation health (0–25 points)
    scores["ndvi_health"] = round(min(ndvi / 0.9, 1.0) * 25, 1)

    # 2. Carbon density vs typology mean (0–20 points)
    ratio = min(carbon_density / typology_mean, 1.0)
    scores["carbon_density_score"] = round(ratio * 20, 1)

    # 3. Scene quality — cloud cover penalty (0–10 points)
    scores["scene_quality"] = round(max(0, (1 - cloud_cover / 100)) * 10, 1)

    # 4. GMW boundary validation (0–20 points)
    scores["location_verified"] = 20 if gmw_validated else 0

    # 5. Restoration fraction (0–15 points)
    scores["restoration_potential"] = round(
        min(max(restoration_fraction, 0), 1) * 15, 1
    )

    # 6. Model confidence (0–10 points)
    scores["model_confidence"] = round(
        min(max(model_confidence, 0), 1) * 10, 1
    )

    total = round(sum(scores.values()), 1)

    if total >= 85:
        grade  = "AAA"
        color  = "#16a34a"
        market = "Premium tier — $45–50 / credit"
    elif total >= 75:
        grade  = "AA"
        color  = "#22c55e"
        market = "High quality — $35–45 / credit"
    elif total >= 65:
        grade  = "A"
        color  = "#84cc16"
        market = "Standard — $25–35 / credit"
    elif total >= 55:
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
