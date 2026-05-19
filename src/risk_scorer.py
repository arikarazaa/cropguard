"""
risk_scorer.py — Maps disease predictions to yield-loss risk estimates.
Bridges the CNN output to the AgroRisk quantification layer.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskAssessment:
    disease: str
    crop: str
    level: str              # None / Low / Moderate / High / Critical
    yield_loss_pct: float   # Estimated % yield loss
    confidence: float       # Model confidence in prediction
    recommendation: str
    time_to_act_hours: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "disease": self.disease,
            "crop": self.crop,
            "risk_level": self.level,
            "yield_loss_pct": self.yield_loss_pct,
            "model_confidence": round(self.confidence * 100, 1),
            "recommendation": self.recommendation,
            "time_to_act_hours": self.time_to_act_hours,
        }

    def __str__(self) -> str:
        return (
            f"[{self.level.upper()}] {self.disease}\n"
            f"  Crop: {self.crop}\n"
            f"  Estimated yield loss: {self.yield_loss_pct:.0f}%\n"
            f"  Model confidence: {self.confidence*100:.1f}%\n"
            f"  Action: {self.recommendation}"
            + (f" (within {self.time_to_act_hours}h)" if self.time_to_act_hours else "")
        )


# ── Risk database ──────────────────────────────────────────────────────────────
# Each entry: (yield_loss_pct, risk_level, recommendation, time_to_act_hours)
_RISK_DB: dict[str, tuple[float, str, str, Optional[int]]] = {
    # Tomato
    "Tomato___Late_blight": (62, "Critical", "Apply copper-based fungicide immediately. Remove heavily infected leaves.", 24),
    "Tomato___Early_blight": (38, "High", "Apply chlorothalonil or mancozeb fungicide. Increase plant spacing.", 48),
    "Tomato___Bacterial_spot": (35, "High", "Apply copper bactericide. Avoid overhead irrigation.", 48),
    "Tomato___Septoria_leaf_spot": (28, "Moderate", "Remove infected leaves. Apply protective fungicide.", 72),
    "Tomato___Leaf_Mold": (20, "Moderate", "Improve ventilation. Apply fungicide if indoors.", 72),
    "Tomato___Target_Spot": (22, "Moderate", "Apply fungicide preventatively. Remove crop debris.", 72),
    "Tomato___Spider_mites_Two-spotted_spider_mite": (18, "Low", "Apply acaricide or neem oil. Increase humidity.", 96),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (80, "Critical", "Remove infected plants immediately. Control whitefly vectors.", 12),
    "Tomato___Tomato_mosaic_virus": (45, "High", "Remove infected plants. Disinfect tools. No chemical cure.", 24),
    "Tomato___healthy": (0, "None", "Healthy crop — continue regular monitoring.", None),

    # Potato
    "Potato___Late_blight": (70, "Critical", "Apply metalaxyl + mancozeb immediately. Destroy infected haulms.", 12),
    "Potato___Early_blight": (40, "High", "Apply chlorothalonil. Ensure adequate potassium nutrition.", 48),
    "Potato___healthy": (0, "None", "Healthy crop — continue regular monitoring.", None),

    # Corn
    "Corn___Northern_Leaf_Blight": (30, "High", "Apply strobilurin fungicide. Plant resistant hybrids next season.", 48),
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot": (24, "Moderate", "Apply triazole fungicide. Improve air circulation.", 72),
    "Corn___Common_rust": (15, "Low", "Monitor spread. Apply fungicide if >50% leaf area affected.", 96),
    "Corn___healthy": (0, "None", "Healthy crop — continue regular monitoring.", None),

    # Apple
    "Apple___Apple_scab": (24, "Moderate", "Apply captan or dodine. Rake and destroy fallen leaves.", 72),
    "Apple___Black_rot": (45, "High", "Prune and destroy infected tissue. Apply copper fungicide.", 48),
    "Apple___Cedar_apple_rust": (20, "Moderate", "Apply myclobutanil. Remove nearby cedar trees if possible.", 72),
    "Apple___healthy": (0, "None", "Healthy crop — continue regular monitoring.", None),

    # Grape
    "Grape___Black_rot": (55, "Critical", "Apply mancozeb or captan. Remove mummified berries.", 24),
    "Grape___Esca_(Black_Measles)": (35, "High", "Prune infected wood. No systemic cure available.", 48),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (20, "Moderate", "Apply copper fungicide preventatively.", 72),
    "Grape___healthy": (0, "None", "Healthy crop — continue regular monitoring.", None),

    # Others — defaults applied for unlisted classes
}

_DEFAULT_RISK = (20, "Moderate", "Consult local agronomist for treatment options.", 72)


def assess_risk(class_name: str, confidence: float) -> RiskAssessment:
    """
    Generate a full RiskAssessment from a predicted disease class and model confidence.

    Args:
        class_name: PlantVillage class string, e.g. 'Tomato___Late_blight'
        confidence: Softmax confidence from model (0–1)

    Returns:
        RiskAssessment dataclass
    """
    parts = class_name.split("___", 1)
    crop = parts[0] if len(parts) == 2 else "Unknown"
    disease_raw = parts[1].replace("_", " ") if len(parts) == 2 else class_name

    yield_loss, level, recommendation, time_to_act = _RISK_DB.get(class_name, _DEFAULT_RISK)

    # Adjust risk level downward if model confidence is low
    if confidence < 0.60 and level != "None":
        level = _downgrade_level(level)
        recommendation = f"[Low confidence — verify manually] {recommendation}"

    return RiskAssessment(
        disease=disease_raw,
        crop=crop,
        level=level,
        yield_loss_pct=yield_loss,
        confidence=confidence,
        recommendation=recommendation,
        time_to_act_hours=time_to_act,
    )


def _downgrade_level(level: str) -> str:
    order = ["None", "Low", "Moderate", "High", "Critical"]
    idx = order.index(level)
    return order[max(0, idx - 1)]


def batch_assess(predictions: list[tuple[str, float]]) -> list[RiskAssessment]:
    """Assess risk for a list of (class_name, confidence) pairs."""
    return [assess_risk(cls, conf) for cls, conf in predictions]


if __name__ == "__main__":
    # Example usage
    test_cases = [
        ("Tomato___Late_blight", 0.961),
        ("Corn___Cercospora_leaf_spot_Gray_leaf_spot", 0.942),
        ("Tomato___healthy", 0.994),
        ("Potato___Late_blight", 0.55),   # Low confidence → downgraded
    ]
    for cls, conf in test_cases:
        ra = assess_risk(cls, conf)
        print(ra)
        print()
