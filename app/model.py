import os
import joblib
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from math import floor
from urllib.parse import quote

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
                # Return this line + next 2 lines as the TLE block
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
def predict_top_events(top_n: int = 4):
    """Return top N upcoming critical events from the CSV dataset."""
    try:
        df = pd.read_csv(CSV_PATH)

        # Parse datetime and probabilities
        df["EPOCH_dt"] = pd.to_datetime(df.iloc[:, 4], errors="coerce", utc=True)
        df["raw_prob"] = pd.to_numeric(df.iloc[:, 5], errors="coerce").fillna(0.0)

        # Normalize probabilities
        max_prob = df["raw_prob"].max()
        df["probability"] = 0.0 if max_prob <= 0 else (df["raw_prob"] / max_prob).clip(0.0, 1.0)

        # Filter: events within next 2 days
        now, cutoff = datetime.now(timezone.utc), datetime.now(timezone.utc) + timedelta(days=2)
        future_df = df[(df["EPOCH_dt"] >= now) & (df["EPOCH_dt"] <= cutoff)]

        if future_df.empty:
            return {"critical_events": [], "status": "info", "message": "No upcoming events found."}

        # Select top N by probability
        critical_df = future_df.sort_values("probability", ascending=False).head(top_n)

        results = []
        for _, row in critical_df.iterrows():
            prob = row["probability"]
            risk_level, maneuver = classify_risk(prob)

            # Satellite & debris names
            sat_name = str(row.iloc[1]) if any(c.isalpha() for c in str(row.iloc[1])) else f"SAT-{row.iloc[1]}"
            debris_name = str(row.iloc[2])

            results.append({
                "satellite": sat_name,
                "satellite_tle": fetch_tle(sat_name),
                "debris": debris_name,
                "debris_tle": fetch_tle(debris_name),
                "tca": row.iloc[4],
                "time_to_impact": time_to_impact(row.iloc[4]),
                "probability": f"{prob*100:.1f}%",
                "risk_level": risk_level,
                "maneuver_suggestion": maneuver,
                "confidence": f"{prob*100:.1f}%"
            })

        return {"critical_events": results, "status": "ok"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
