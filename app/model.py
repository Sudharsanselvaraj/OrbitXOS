import os
import joblib
import pandas as pd
from datetime import datetime, timezone
from math import floor

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "prop_risk_model_resaved.joblib")
CSV_PATH = os.path.join(os.path.dirname(__file__), "reduced_file2.csv")

# Load model (optional; will only be used if CSV has no probabilities)
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# --- Risk classification ---
def classify_risk(prob: float):
    if prob < 0.3:
        return "Low", "No action needed"
    elif prob < 0.6:
        return "Medium", "Monitor, prepare retrograde burn"
    elif prob < 0.8:
        return "High", "Plan radial maneuver"
    else:
        return "Critical", "Execute immediate retrograde burn"

# --- Time to impact ---
def time_to_impact(tca_str: str) -> str:
    try:
        tca = datetime.fromisoformat(tca_str)
        if tca.tzinfo is None:
            tca = tca.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        delta_sec = max(0, (tca - now).total_seconds())

        days = floor(delta_sec // 86400)
        hours = floor((delta_sec % 86400) // 3600)
        minutes = floor((delta_sec % 3600) // 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        else:
            return f"{hours}h {minutes}m"
    except Exception:
        return "N/A"

# --- Predict top events ---
def predict_top_events(top_n: int = 4):
    try:
        df = pd.read_csv(CSV_PATH)
        df['EPOCH_dt'] = pd.to_datetime(df.iloc[:, 3], errors='coerce', utc=True)
        now = datetime.now(timezone.utc)
        future_df = df[df['EPOCH_dt'] > now]

        if future_df.empty:
            return {"critical_events": [], "status": "info", "message": "No upcoming events found."}

        # --- Check if probability column exists in CSV ---
        if future_df.shape[1] > 4:  # assume column 4 contains probability
            future_df["probability"] = future_df.iloc[:, 4].astype(float)
        elif model is not None:
            # Fall back to model prediction
            cols = future_df.columns
            future_df['i_ecc']  = future_df['ECCENTRICITY'] if 'ECCENTRICITY' in cols else 0.0
            future_df['j_ecc']  = future_df['ECCENTRICITY'] if 'ECCENTRICITY' in cols else 0.0
            future_df['i_incl'] = future_df['INCLINATION']  if 'INCLINATION'  in cols else 0.0
            future_df['j_incl'] = future_df['INCLINATION']  if 'INCLINATION'  in cols else 0.0
            future_df['i_sma']  = future_df['MEAN_MOTION']  if 'MEAN_MOTION'  in cols else 0.0
            future_df['j_sma']  = future_df['MEAN_MOTION']  if 'MEAN_MOTION'  in cols else 0.0
            future_df['vrel_kms'] = future_df['VREL_KMS'] if 'VREL_KMS' in cols else 0.5

            feature_cols = ["i_ecc", "j_ecc", "i_incl", "j_incl", "i_sma", "j_sma", "vrel_kms"]
            for c in feature_cols:
                if c not in future_df.columns:
                    future_df[c] = 0.0

            X = future_df[feature_cols]
            probs = model.predict_proba(X)[:, 1]
            future_df["probability"] = probs
        else:
            return {"status": "error", "message": "No probability column and model not available."}

        # Sort and take top_n
        critical_df = future_df.sort_values("probability", ascending=False).head(top_n)

        results = []
        for _, row in critical_df.iterrows():
            risk_level, maneuver = classify_risk(row["probability"])
            results.append({
                "satellite": row.iloc[1],
                "debris": row.iloc[2],
                "tca": row.iloc[3],
                "time_to_impact": time_to_impact(row.iloc[3]),
                "probability": f"{row['probability']*100:.1f}%",
                "risk_level": risk_level,
                "maneuver_suggestion": maneuver,
                "confidence": f"{row['probability']*100:.1f}%"
            })

        return {"critical_events": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}
