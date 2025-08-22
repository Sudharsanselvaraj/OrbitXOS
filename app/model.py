import os
import joblib
import pandas as pd
from datetime import datetime, timezone, timedelta
from math import floor
import math
from typing import List, Dict, Tuple
from sgp4.api import Satrec, jday

# -------------------------------
# Paths & Model
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "prop_risk_model_resaved.joblib")
CSV_PATH = os.path.join(BASE_DIR, "reduced_file2.csv")
TLE_FILE = os.path.join(BASE_DIR, "active_satellites_tle.txt")

# Load risk model
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# -------------------------------
# Helper functions
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
        return f"{days}d {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"
    except Exception:
        return "N/A"

def normalize_tle_block(tle_text: str) -> Tuple[str, str, str]:
    lines = [ln.strip() for ln in tle_text.strip().splitlines() if ln.strip()]
    if len(lines) >= 3 and lines[1].startswith("1 ") and lines[2].startswith("2 "):
        return lines[0], lines[1], lines[2]
    if len(lines) >= 2 and lines[0].startswith("1 ") and lines[1].startswith("2 "):
        return "UNKNOWN", lines[0], lines[1]
    raise ValueError("Invalid TLE format")

def fetch_tle(name: str) -> str:
    try:
        with open(TLE_FILE, "r") as f:
            lines = f.readlines()
        name_upper = name.upper().strip()
        for i, line in enumerate(lines):
            if line.strip().upper() == name_upper:
                if i + 2 < len(lines):
                    return "\n".join([lines[i].strip(), lines[i+1].strip(), lines[i+2].strip()])
                else:
                    return ""
        return ""
    except Exception:
        return ""

def is_leo(tle_block: str) -> bool:
    try:
        _, _, L2 = normalize_tle_block(tle_block)
        mm = float(L2[52:63])
        return mm > 10.0
    except Exception:
        return False

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
# Predict top events (LEO only)
# -------------------------------
def predict_top_events(top_n: int = 4) -> Dict:
    try:
        df = pd.read_csv(CSV_PATH)
        df["EPOCH_dt"] = pd.to_datetime(df["tca"], errors="coerce", utc=True)

        # Compute probability using model or fallback
        if model:
            try:
                feature_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else df.columns[:model.n_features_in_]
                features = df[feature_cols].fillna(0.0)
                df["raw_prob"] = model.predict_proba(features)[:,1]
            except Exception:
                df["raw_prob"] = 1 / (1 + df["miss_km"].astype(float))
        else:
            df["raw_prob"] = 1 / (1 + df["miss_km"].astype(float))

        max_prob = df["raw_prob"].max()
        df["probability"] = 0.0 if max_prob <= 0 else (df["raw_prob"] / max_prob).clip(0.0, 1.0)

        # Filter for upcoming 2 days
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=2)
        df = df[(df["EPOCH_dt"] >= now) & (df["EPOCH_dt"] <= cutoff)]

        if df.empty:
            return {"critical_events": [], "status": "info", "message": "No upcoming events found."}

        results = []
        for _, row in df.sort_values("probability", ascending=False).head(top_n).iterrows():
            sat_name = str(row["i_name"])
            debris_name = str(row["j_name"])
            sat_tle = fetch_tle(sat_name)
            debris_tle = fetch_tle(debris_name)

            # Skip if satellite TLE is not LEO
            if not is_leo(sat_tle):
                continue

            prob = row["probability"]
            risk_level, maneuver = classify_risk(prob)

            results.append({
                "satellite": sat_name,
                "satellite_tle": sat_tle,
                "debris": debris_name,
                "debris_tle": debris_tle,
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
