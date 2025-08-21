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

        # Convert EPOCH to datetime (UTC)
        df['EPOCH_dt'] = pd.to_datetime(df['EPOCH'], errors='coerce', utc=True)
        now = datetime.now(timezone.utc)

        # Filter only future events
        future_df = df[df['EPOCH_dt'] > now]

        # If not enough future events, pick closest upcoming events regardless
        if len(future_df) < top_n:
            future_df = df[df['EPOCH_dt'].notna()].sort_values("EPOCH_dt").head(top_n)

        # Map CSV columns to model features
        df_cols = df.columns
        df['i_ecc'] = df['ECCENTRICITY'] if 'ECCENTRICITY' in df_cols else 0.0
        df['j_ecc'] = df['ECCENTRICITY'] if 'ECCENTRICITY' in df_cols else 0.0
        df['i_incl'] = df['INCLINATION'] if 'INCLINATION' in df_cols else 0.0
        df['j_incl'] = df['INCLINATION'] if 'INCLINATION' in df_cols else 0.0
        df['i_sma'] = df['MEAN_MOTION'] if 'MEAN_MOTION' in df_cols else 0.0
        df['j_sma'] = df['MEAN_MOTION'] if 'MEAN_MOTION' in df_cols else 0.0
        df['vrel_kms'] = df['VREL_KMS'] if 'VREL_KMS' in df_cols else 0.5

        feature_cols = ["i_ecc", "j_ecc", "i_incl", "j_incl", "i_sma", "j_sma", "vrel_kms"]
        X = future_df[feature_cols]

        # Predict probabilities
        probs = model.predict_proba(X)[:, 1]
        future_df["probability"] = probs

        # Get top-N critical events
        critical_df = future_df.sort_values("probability", ascending=False).head(top_n)

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

