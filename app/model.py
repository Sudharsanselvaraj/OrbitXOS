import os
import joblib
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from math import floor
from urllib.parse import quote

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "prop_risk_model_resaved.joblib")
CSV_PATH = os.path.join(BASE_DIR, "reduced_file2.csv")

# Load model (optional, fallback if no probs in CSV)
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


# -------------------------------
# Risk classification
# -------------------------------
def classify_risk(prob: float):
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


# -------------------------------
# Fetch TLE block from CelesTrak
# -------------------------------
def fetch_tle(name: str) -> str:
    """
    Fetch full TLE block (name + line1 + line2) from CelesTrak.
    Returns the block as a single string with line breaks.
    """
    try:
        url = f"https://celestrak.org/NORAD/elements/gp.php?NAME={quote(name)}&FORMAT=tle"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        tle_text = resp.text.strip()

        # Ensure it's the expected 3-line block
        lines = tle_text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[:3])
        else:
            return tle_text  # fallback (incomplete TLE)
    except Exception as e:
        return f"TLE fetch failed: {e}"


# -------------------------------
# Predict top events
# -------------------------------
def predict_top_events(top_n: int = 4):
    try:
        df = pd.read_csv(CSV_PATH)

        # Epoch (column 4), raw risk score (column 5)
        df["EPOCH_dt"] = pd.to_datetime(df.iloc[:, 4], errors="coerce", utc=True)
        df["raw_prob"] = pd.to_numeric(df.iloc[:, 5], errors="coerce").fillna(0.0)

        # Normalize probabilities
        max_prob = df["raw_prob"].max()
        df["probability"] = (
            0.0 if max_prob <= 0 else (df["raw_prob"] / max_prob).clip(0.0, 1.0)
        )

        # Filter: only next 2 days
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=2)
        future_df = df[(df["EPOCH_dt"] >= now) & (df["EPOCH_dt"] <= cutoff)]

        if future_df.empty:
            return {
                "critical_events": [],
                "status": "info",
                "message": "No upcoming events found.",
            }

        # Sort by probability & take top N
        critical_df = future_df.sort_values("probability", ascending=False).head(top_n)

        results = []
        for _, row in critical_df.iterrows():
            prob = row["probability"]
            risk_level, maneuver = classify_risk(prob)

            # Satellite & debris names
            sat_name = str(row.iloc[1])
            if not any(c.isalpha() for c in sat_name):
                sat_name = f"SAT-{sat_name}"  # fallback

            debris_name = str(row.iloc[2])

            # Fetch TLEs
            sat_tle = fetch_tle(sat_name)
            debris_tle = fetch_tle(debris_name)

            results.append(
                {
                    "satellite": sat_name,
                    "satellite_tle": sat_tle,
                    "debris": debris_name,
                    "debris_tle": debris_tle,
                    "tca": row.iloc[4],
                    "time_to_impact": time_to_impact(row.iloc[4]),
                    "probability": f"{prob*100:.1f}%",
                    "risk_level": risk_level,
                    "maneuver_suggestion": maneuver,
                    "confidence": f"{prob*100:.1f}%",
                }
            )

        return {"critical_events": results, "status": "ok"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
