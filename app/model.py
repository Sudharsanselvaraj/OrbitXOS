import os
import joblib
import pandas as pd
from datetime import datetime, timezone

# Load model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "prop_risk_model_resaved.joblib")
CSV_PATH = os.path.join(os.path.dirname(__file__), "active_tle_catalog.csv")
model = joblib.load(MODEL_PATH)


# --- Replace classify_risk with this (no special symbols, ops terms) ---
def classify_risk(prob: float):
    if prob < 0.3:
        return "Low", "No action needed"
    elif prob < 0.6:
        return "Medium", "Monitor, prepare retrograde burn"
    elif prob < 0.8:
        return "High", "Plan radial maneuver"
    else:
        return "Critical", "Execute immediate retrograde burn"



# --- Replace time_to_impact with this (never negative, clean output) ---
from math import floor

def time_to_impact(tca_str: str) -> str:
    """Return countdown until TCA; clamped at 0 if already passed."""
    try:
        tca = datetime.fromisoformat(tca_str)
        if tca.tzinfo is None:
            tca = tca.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        delta_sec = max(0, (tca - now).total_seconds())  # clamp negatives

        days = floor(delta_sec // 86400)
        hours = floor((delta_sec % 86400) // 3600)
        minutes = floor((delta_sec % 3600) // 60)

        # Show the most useful parts; omit zero days for brevity
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        else:
            return f"{hours}h {minutes}m"
    except Exception:
        return "N/A"



# --- Replace predict_top_events with this (strictly future-only) ---
def predict_top_events(top_n: int = 4):
    try:
        df = pd.read_csv(CSV_PATH)

        # Parse EPOCH and filter strictly to future events
        df['EPOCH_dt'] = pd.to_datetime(df['EPOCH'], errors='coerce', utc=True)
        now = datetime.now(timezone.utc)
        future_df = df[df['EPOCH_dt'] > now].copy()  # copy to avoid chained-assign issues

        # If there are no future events, return empty list (but not past)
        if future_df.empty:
            return {"critical_events": [], "status": "info", "message": "No upcoming events found."}

        # Build feature columns on the filtered frame
        cols = future_df.columns
        future_df['i_ecc']  = future_df['ECCENTRICITY'] if 'ECCENTRICITY' in cols else 0.0
        future_df['j_ecc']  = future_df['ECCENTRICITY'] if 'ECCENTRICITY' in cols else 0.0
        future_df['i_incl'] = future_df['INCLINATION']  if 'INCLINATION'  in cols else 0.0
        future_df['j_incl'] = future_df['INCLINATION']  if 'INCLINATION'  in cols else 0.0
        future_df['i_sma']  = future_df['MEAN_MOTION']  if 'MEAN_MOTION'  in cols else 0.0
        future_df['j_sma']  = future_df['MEAN_MOTION']  if 'MEAN_MOTION'  in cols else 0.0
        future_df['vrel_kms'] = future_df['VREL_KMS']   if 'VREL_KMS'     in cols else 0.5  # placeholder

        feature_cols = ["i_ecc", "j_ecc", "i_incl", "j_incl", "i_sma", "j_sma", "vrel_kms"]

        # Ensure all feature columns exist (fallback to zeros if missing)
        for c in feature_cols:
            if c not in future_df.columns:
                future_df[c] = 0.0

        X = future_df[feature_cols]

        # Predict probabilities
        probs = model.predict_proba(X)[:, 1]
        future_df["probability"] = probs

        # Select up to top_n (only future rows)
        critical_df = future_df.sort_values("probability", ascending=False).head(top_n)

        results = []
        for _, row in critical_df.iterrows():
            risk_level, maneuver = classify_risk(float(row["probability"]))
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

