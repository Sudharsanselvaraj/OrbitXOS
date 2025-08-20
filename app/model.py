import joblib
import numpy as np
import os
import pandas as pd
from datetime import datetime, timezone

# -------------------------
# Load trained model
# -------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "prop_risk_model_resaved.joblib")
model = joblib.load(MODEL_PATH)

# -------------------------
# Load TLE catalog for metadata
# (must have OBJECT_ID, OBJECT_NAME, TCA, etc. in CSV)
# -------------------------
CSV_PATH = os.path.join(os.path.dirname(__file__), "active_tle_catalog.csv")
tle_catalog = pd.read_csv(CSV_PATH)

def get_object_name(object_id):
    """Fetch satellite/debris name from catalog."""
    row = tle_catalog.loc[tle_catalog["OBJECT_ID"] == object_id]
    if not row.empty:
        return row.iloc[0]["OBJECT_NAME"]
    return f"UNKNOWN-{object_id}"

def get_tca(object_id):
    """Fetch TCA (time of closest approach) if available."""
    row = tle_catalog.loc[tle_catalog["OBJECT_ID"] == object_id]
    if not row.empty and "TCA" in row.columns:
        return row.iloc[0]["TCA"]
    return datetime.now(timezone.utc).isoformat()

def compute_time_to_impact(tca_str):
    """Compute time difference from now to TCA."""
    try:
        tca = datetime.fromisoformat(tca_str.replace("Z", "+00:00"))
        delta = tca - datetime.now(timezone.utc)
        hrs, rem = divmod(int(delta.total_seconds()), 3600)
        mins = rem // 60
        sign = "-" if delta.total_seconds() < 0 else "+"
        return f"{sign}{abs(hrs)}h {abs(mins)}m"
    except Exception:
        return "UNKNOWN"

# -------------------------
# Prediction
# -------------------------
def predict_risk(i_ecc, j_ecc, i_incl, j_incl, i_sma, j_sma, vrel_kms,
                 sat_id=None, debris_id=None):
    """
    Run risk prediction for one satellite-debris conjunction.
    """
    # 7 features only (matching trained model)
    X = np.array([[i_ecc, j_ecc, i_incl, j_incl, i_sma, j_sma, vrel_kms]])
    prob = model.predict_proba(X)[0][1]

    # Risk classification
    if prob >= 0.6:
        risk = "High"
        maneuver = "Prepare avoidance, ΔV radial, ~0.2-0.5 m/s"
    elif prob >= 0.3:
        risk = "Medium"
        maneuver = "Monitor closely, potential maneuver if probability increases"
    else:
        risk = "Low"
        maneuver = "No immediate action required"

    # Attach metadata
    sat_name = get_object_name(sat_id) if sat_id else "UNKNOWN"
    deb_name = get_object_name(debris_id) if debris_id else "UNKNOWN"
    tca = get_tca(sat_id)
    tti = compute_time_to_impact(tca)

    return {
        "satellite": sat_name,
        "debris": deb_name,
        "tca": tca,
        "time_to_impact": tti,
        "probability": f"{int(prob*100)}%",
        "risk_level": risk,
        "maneuver_suggestion": maneuver,
        "confidence": f"{int(prob*100)}%"
    }
