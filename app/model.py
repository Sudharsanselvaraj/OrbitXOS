import os
import joblib
import pandas as pd
from datetime import datetime, timezone, timedelta
from math import floor

# -------------------------------
# Paths & Model
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "prop_risk_model_resaved.joblib")
CSV_PATH = os.path.join(BASE_DIR, "reduced_file2.csv")
TLE_FILE = os.path.join(BASE_DIR, "active_satellites_tle.txt")

# Load risk model if available
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# -------------------------------
# Risk classification
# -------------------------------
def classify_risk(prob: float):
    """Classify probability into risk level and suggest maneuver."""
    if prob < 0.3:
        return "Low", "No action needed"
    elif prob < 0.6:
        return "Medium", "Monitor, prepare retrograde burn"
    elif prob < 0.8:
        return "High", "Plan radial maneuver"
    else:
        return "Critical", "Execute immediate retrograde burn"

# -------------------------------
# Time to impact
# -------------------------------
def time_to_impact(tca_str: str) -> str:
    """Return time until TCA in days/hours/minutes format."""
    try:
        tca = datetime.fromisoformat(tca_str)
        if tca.tzinfo is None:
            tca = tca.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta_sec = max(0, (tca - now).total_seconds())
        days = floor(delta_sec // 86400)
        hours = floor((delta_sec % 86400) // 3600)
        minutes = floor((delta_sec % 3600) // 60)
        return f"{days}d {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"
    except Exception:
        return "N/A"

# -------------------------------
# Fetch TLE block from local file
# -------------------------------
def fetch_tle(name: str) -> str:
    """
    Fetch full TLE block (name + line1 + line2) from the local active_satellites_tle.txt file.
    Returns the block as a single 3-line string. Returns a message if not found.
    """
    try:
        with open(TLE_FILE, "r") as f:
            lines = f.readlines()

        name_upper = name.upper().strip()
        for i in range(len(lines)):
            line = lines[i].strip()
            if line.upper() == name_upper:
                if i + 2 < len(lines):
                    return "\n".join([lines[i].strip(), lines[i + 1].strip(), lines[i + 2].strip()])
                else:
                    return "TLE block incomplete in file"
        return f"TLE not found for '{name}' in local file"
    except Exception as e:
        return f"Failed to read TLE file: {e}"

# -------------------------------
# Predict Top Events
# -------------------------------
def predict_top_events(top_n: int = 6):
    """Return top N upcoming critical events (pads with past/low events if needed)."""
    try:
        df = pd.read_csv(CSV_PATH)

        # Parse datetime
        df["EPOCH_dt"] = pd.to_datetime(df["tca"], errors="coerce", utc=True)

        # -------------------------------
        # Probability calculation
        # -------------------------------
        if model:
            try:
                if hasattr(model, "feature_names_in_"):
                    feature_cols = list(model.feature_names_in_)
                else:
                    feature_cols = df.columns[:model.n_features_in_]

                features = df[feature_cols].fillna(0.0)
                df["raw_prob"] = model.predict_proba(features)[:, 1]
            except Exception as e:
                print("⚠ Model input mismatch, fallback to heuristic:", e)
                df["raw_prob"] = 1 / (1 + df["miss_km"].astype(float))
        else:
            df["raw_prob"] = 1 / (1 + df["miss_km"].astype(float))

        # Normalize probabilities
        max_prob = df["raw_prob"].max()
        df["probability"] = 0.0 if max_prob <= 0 else (df["raw_prob"] / max_prob).clip(0.0, 1.0)

        # -------------------------------
        # Select events
        # -------------------------------
        now, cutoff = datetime.now(timezone.utc), datetime.now(timezone.utc) + timedelta(days=2)
        future_df = df[(df["EPOCH_dt"] >= now) & (df["EPOCH_dt"] <= cutoff)]

        # Sort by probability
        future_df = future_df.sort_values("probability", ascending=False)

        # If fewer than top_n events, pad with other low-probability / past events
        if len(future_df) < top_n:
            filler_df = df.drop(future_df.index).sort_values("probability", ascending=False)
            combined_df = pd.concat([future_df, filler_df]).head(top_n)
        else:
            combined_df = future_df.head(top_n)

        # -------------------------------
        # Build results
        # -------------------------------
        results = []
        for _, row in combined_df.iterrows():
            prob = row["probability"]
            risk_level, maneuver = classify_risk(prob)

            sat_name = str(row["i_name"])
            debris_name = str(row["j_name"])

            results.append({
                "satellite": sat_name,
                "satellite_tle": fetch_tle(sat_name),
                "debris": debris_name,
                "debris_tle": fetch_tle(debris_name),
                "tca": row["tca"],
                "time_to_impact": time_to_impact(row["tca"]),
                "probability": f"{prob*100:.1f}%",
                "risk_level": risk_level,
                "maneuver_suggestion": maneuver,
                "confidence": f"{prob*100:.1f}%"
            })

        return {"critical_events": results, "status": "ok"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
