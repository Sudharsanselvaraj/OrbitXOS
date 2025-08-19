import joblib
import os
import pandas as pd

# Path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "prop_risk_model.joblib")

# Load the trained GradientBoostingClassifier
try:
    risk_model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Could not load model from {MODEL_PATH}: {e}")

def predict_risk(i_ecc: float, j_ecc: float, i_incl: float, j_incl: float, vrel_kms: float):
    """
    Takes orbital features and returns collision risk probability, risk level,
    and maneuver suggestion.
    """
    features = pd.DataFrame([{
        "i_ecc": i_ecc,
        "j_ecc": j_ecc,
        "i_incl": i_incl,
        "j_incl": j_incl,
        "vrel_kms": vrel_kms
    }])

    # Predict risk probability (class 1)
    prob = risk_model.predict_proba(features)[0][1]

    # Interpret risk
    if prob > 0.7:
        level = "High"
        suggestion = "Prepare avoidance, ΔV radial, ~0.2–0.5 m/s"
    elif prob > 0.4:
        level = "Medium"
        suggestion = "Monitor & consider small avoidance maneuver"
    else:
        level = "Low"
        suggestion = "No immediate action required"

    return {
        "probability": round(float(prob), 3),
        "risk_level": level,
        "maneuver_suggestion": suggestion
    }

