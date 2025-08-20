import os
import joblib
import pandas as pd
from datetime import datetime, timezone

# Load model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "prop_risk_model_resaved.joblib")
CSV_PATH = os.path.join(os.path.dirname(__file__), "active_tle_catalog.csv")
model = joblib.load(MODEL_PATH)


def classify_risk(prob: float):
    if prob < 0.3:
        return "Low", "No immediate action required"
    elif prob < 0.6:
        return "Medium", "Monitor closely, prepare possible maneuver"
    elif prob < 0.8:
        return "High", "Prepare avoidance, ΔV radial, ~0.2–0.5 m/s"
    else:
        return "Critical", "Immediate avoidance required, ΔV > 0.5 m/s"


def time_to_impact(tca_str: str) -> str:
    """Return human-readable time until (or since) TCA."""
    try:
        # Parse datetime
        tca = datetime.fromisoformat(tca_str)
        # If naive (no timezone), assume UTC
        if tca.tzinfo is None:
            tca = tca.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        delta = tca - now
        hours, remainder = divmod(abs(delta.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        sign = "-" if delta.total_seconds() < 0 else "+"
        return f"{sign}{int(abs(delta.days))}d {int(hours)%24}h {int(minutes)}m"
    except Exception:
        return "N/A"



def predict_top_events(top_n: int = 4):
    try:
        df = pd.read_csv(CSV_PATH)

        # Map CSV columns to model features
        if 'ECCENTRICITY' in df.columns:
            df['i_ecc'] = df['ECCENTRICITY']
            df['j_ecc'] = df['ECCENTRICITY']
        else:
            df['i_ecc'] = 0.0
            df['j_ecc'] = 0.0

        if 'INCLINATION' in df.columns:
            df['i_incl'] = df['INCLINATION']
            df['j_incl'] = df['INCLINATION']
        else:
            df['i_incl'] = 0.0
            df['j_incl'] = 0.0

        if 'MEAN_MOTION' in df.columns:
            df['i_sma'] = df['MEAN_MOTION']
            df['j_sma'] = df['MEAN_MOTION']
        else:
            df['i_sma'] = 0.0
            df['j_sma'] = 0.0

        # vrel_kms placeholder (or compute if available)
        if 'VREL_KMS' not in df.columns:
            df['vrel_kms'] = 0.5  # default value

        # Features for prediction
        feature_cols = ["i_ecc", "j_ecc", "i_incl", "j_incl", "i_sma", "j_sma", "vrel_kms"]
        X = df[feature_cols]

        # Predict probabilities
        probs = model.predict_proba(X)[:, 1]
        df["probability"] = probs

        # Get top-N critical events
        critical_df = df.sort_values("probability", ascending=False).head(top_n)

        results = []
        for _, row in critical_df.iterrows():
            risk_level, maneuver = classify_risk(row["probability"])
            results.append({
                "satellite": row.get("OBJECT_NAME", "Unknown"),
                "debris": row.get("OBJECT_ID", "Unknown"),
                "tca": row.get("EPOCH", "N/A"),
                "time_to_impact": time_to_impact(row.get("EPOCH", "N/A")),
                "probability": f"{row['probability']*100:.1f}%",
                "risk_level": risk_level,
                "maneuver_suggestion": maneuver,
                "confidence": f"{row['probability']*100:.1f}%"
            })

        return {"critical_events": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}
