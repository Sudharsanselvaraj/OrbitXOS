import joblib
import numpy as np
import os

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "prop_risk_model_resaved.joblib")
model = joblib.load(MODEL_PATH)

def predict_risk(i_ecc, j_ecc, i_incl, j_incl, i_sma, j_sma, vrel_kms, i_raan, j_raan):
    """
    Run risk prediction for one satellite-debris conjunction.
    Returns probability, risk level, and maneuver suggestion.
    """

    # Features in the correct order (7 features expected by GradientBoostingClassifier)
    X = np.array([[i_ecc, j_ecc, i_incl, j_incl, i_sma, j_sma, vrel_kms]])

    # Predict probability
    prob = model.predict_proba(X)[0][1]

    # Define thresholds
    if prob >= 0.6:
        risk = "High"
        maneuver = "Prepare avoidance, ΔV radial, ~0.2-0.5 m/s"
    elif prob >= 0.3:
        risk = "Medium"
        maneuver = "Monitor closely, potential maneuver if probability increases"
    else:
        risk = "Low"
        maneuver = "No immediate action required"

    return {
        "probability": float(round(prob, 3)),  # e.g. 0.692
        "risk_level": risk,
        "maneuver_suggestion": maneuver
    }
