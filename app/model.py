import os
import joblib
import pandas as pd
from datetime import datetime, timezone
# Load model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "prop_risk_model_resaved.joblib")
CSV_PATH = os.path.join(os.path.dirname(__file__), "active_tle_catalog.csv")
model = joblib.load(MODEL_PATH)


def classify_risk(prob: float):
    """Convert probability (0–1) into risk level + maneuver suggestion."""
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
        tca = datetime.fromisoformat(tca_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = tca - now
        hours, remainder = divmod(abs(delta).seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        sign = "-" if delta.total_seconds() < 0 else "+"
        return f"{sign}{abs(delta.days)}d {hours}h {minutes}m"
    except Exception:
        return "N/A"


def predict_top_events(top_n: int = 4):
    """Load catalog, predict risks, return top-N most critical events."""
    try:
        df = pd.read_csv(CSV_PATH)

        # Expected feature columns for the model
        expected_features = ["i_ecc", "j_ecc", "i_incl", "j_incl", "i_sma", "j_sma", "vrel_kms"]

        # Check which features actually exist in the CSV
        feature_cols = [col for col in expected_features if col in df.columns]

        if not feature_cols:
            return {"status": "error", "message": "None of the required feature columns are present in the CSV."}

        # Predict probabilities
        X = df[feature_cols]
        probs = model.predict_proba(X)[:, 1]

        # Attach predictions
        df["probability"] = probs

        # Sort & pick top N
        critical_df = df.sort_values("probability", ascending=False).head(top_n)

        results = []
        for _, row in critical_df.iterrows():
            risk_level, maneuver = classify_risk(row["probability"])
            results.append({
                "satellite": row.get("satellite", "Unknown"),
                "debris": row.get("debris", "Unknown"),
                "tca": row.get("tca", "N/A"),
                "time_to_impact": time_to_impact(row.get("tca", "N/A")),
                "probability": f"{row['probability']*100:.1f}%",
                "risk_level": risk_level,
                "maneuver_suggestion": maneuver,
                "confidence": f"{row['probability']*100:.1f}%"
            })

        return {"critical_events": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}
