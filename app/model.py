import os
import joblib
import pandas as pd
from datetime import datetime, timezone, timedelta
from math import floor

# --- Paths ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "prop_risk_model_resaved.joblib")
CSV_PATH = os.path.join(os.path.dirname(__file__), "reduced_file2.csv")

# --- Load model (optional, fallback if no probability column) ---
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

        # --- Adjust column names according to your CSV ---
        # Example CSV columns:
        # 0:i_index, 1:Satellite, 2:Debris, 3:TCA, 4:RawProbability, ...
        sat_col = df.columns[1]  # satellite name
        debris_col = df.columns[2]  # debris name
        tca_col = df.columns[3]  # TCA datetime
        prob_col = df.columns[4]  # raw probability/risk score

        # Convert TCA to datetime
        df['EPOCH_dt'] = pd.to_datetime(df[tca_col], errors='coerce', utc=True)

        # Convert raw probability to numeric
        df['raw_prob'] = pd.to_numeric(df[prob_col], errors='coerce').fillna(0.0)

        # --- Dynamic scaling of probabilities ---
        max_prob = df['raw_prob'].max()
        if max_prob <= 0:
            df['probability'] = 0.0
        else:
            df['probability'] = df['raw_prob'] / max_prob  # scale 0–1 proportionally
        df['probability'] = df['probability'].clip(0.0, 1.0)

        # --- Filter events: today + next day ---
        now = datetime.now(timezone.utc)
        tomorrow = now + timedelta(days=2)
        future_df = df[(df['EPOCH_dt'] >= now) & (df['EPOCH_dt'] <= tomorrow)]

        if future_df.empty:
            return {"critical_events": [], "status": "info", "message": "No upcoming events found."}

        # --- Sort by probability descending ---
        critical_df = future_df.sort_values("probability", ascending=False).head(top_n)

        results = []
        for _, row in critical_df.iterrows():
            prob = row['probability']
            risk_level, maneuver = classify_risk(prob)
            results.append({
                "satellite": row[sat_col],
                "debris": row[debris_col],
                "tca": row[tca_col],
                "time_to_impact": time_to_impact(row[tca_col]),
                "probability": f"{prob*100:.1f}%",
                "risk_level": risk_level,
                "maneuver_suggestion": maneuver,
                "confidence": f"{prob*100:.1f}%"
            })

        return {"critical_events": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}
