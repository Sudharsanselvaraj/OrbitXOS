import os
import math
import joblib
import pandas as pd
from datetime import datetime, timezone
from math import floor
from typing import Dict, List, Tuple, Optional

# -------------------------------
# Paths & Optional Model
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "prop_risk_model_resaved.joblib")
CSV_PATH = os.path.join(BASE_DIR, "reduced_file2.csv")
TLE_FILE = os.path.join(BASE_DIR, "active_satellites_tle.txt")

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


# -------------------------------
# Utilities
# -------------------------------
def _normalize_key(s: str) -> str:
    """Uppercase + collapse internal whitespace for stable matching."""
    return " ".join(str(s).strip().upper().split())


def _build_tle_index(tle_path: str) -> Dict[str, str]:
    """
    Index active_satellites_tle.txt into a dict:
      KEY: normalized name
      VAL: 3-line TLE block
    Assumes file is a sequence of 3-line blocks (name, L1, L2).
    """
    idx: Dict[str, str] = {}
    if not os.path.exists(tle_path):
        return idx

    with open(tle_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]

    i = 0
    n = len(lines)
    while i + 2 < n:
        name = lines[i].strip()
        l1 = lines[i + 1].strip() if i + 1 < n else ""
        l2 = lines[i + 2].strip() if i + 2 < n else ""
        # Basic sanity: TLE lines should start with "1 " and "2 "
        if name and l1.startswith("1 ") and l2.startswith("2 "):
            key = _normalize_key(name)
            idx[key] = f"{name}\n{l1}\n{l2}"
            i += 3
        else:
            # If misaligned, advance by 1 and try again (tolerant parser)
            i += 1
    return idx


# Build the TLE index once
_TLE_INDEX = _build_tle_index(TLE_FILE)


def fetch_tle(name: str) -> str:
    """
    Fetch a 3-line TLE block by name from the local file index.
    1) Exact (normalized) match
    2) Contains fallback (if exact not found)
    """
    if not name:
        return "TLE not found (empty name)"

    key = _normalize_key(name)
    if key in _TLE_INDEX:
        return _TLE_INDEX[key]

    # Fallback: substring search over keys (for slight naming variants)
    # e.g., "YAOGAN-17 01A" vs "YAOGAN-17 01A (SOMETHING)"
    for k, block in _TLE_INDEX.items():
        if key in k or k in key:
            return block

    return f"TLE not found for '{name}' in local file"


def time_to_impact(tca_str: str) -> str:
    """Return time until TCA in days/hours/minutes format from now (UTC)."""
    try:
        tca = datetime.fromisoformat(str(tca_str))
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


def classify_risk(prob: float) -> Tuple[str, str]:
    """Map probability (0–1) to risk label + maneuver suggestion."""
    if prob < 0.3:
        return "Low", "No action needed"
    elif prob < 0.6:
        return "Medium", "Monitor, prepare retrograde burn"
    elif prob < 0.8:
        return "High", "Plan radial maneuver"
    else:
        return "Critical", "Execute immediate retrograde burn"


# -------------------------------
# Probability model (heuristic)
# -------------------------------
def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def compute_probability(miss_km: float, vrel_kms: float) -> float:
    """
    Heuristic probability in [0,1] using:
      - miss distance (smaller -> higher risk)
      - relative velocity (faster -> higher risk)
    Tuned to avoid everything becoming 100%.
    """
    # Base risk from miss distance (<=0.1 km ~ very close)
    base = 1.0 / (1.0 + miss_km)            # 0.5 at 1 km, ~0.91 at 0.1 km

    # Normalize velocity roughly: ~10 km/s typical LEO, cap at 2x
    vel_factor = min(vrel_kms / 10.0, 2.0)  # 0..2 range

    # Combine and scale down to reduce saturation
    prob = base * (0.5 + 0.5 * vel_factor)  # 0.5..1.5x multiplier
    prob *= 0.6                             # global scaling to avoid 1.0 saturation

    # Soft cap
    return max(0.0, min(prob, 0.98))


def _model_probability_block(df: pd.DataFrame) -> pd.Series:
    """
    Use sklearn model if available AND compatible.
    Fallback to heuristic otherwise.
    """
    if model is None:
        return df.apply(lambda r: compute_probability(_safe_float(r.get("miss_km")),
                                                      _safe_float(r.get("vrel_kms"))), axis=1)

    try:
        # Best case: model exposes feature names
        if hasattr(model, "feature_names_in_"):
            feats = list(model.feature_names_in_)
            X = df[feats].copy()
            X = X.fillna(0.0)
            proba = model.predict_proba(X)[:, 1]
            # blend a bit with heuristic to avoid saturation & keep variability
            heur = df.apply(lambda r: compute_probability(_safe_float(r.get("miss_km")),
                                                          _safe_float(r.get("vrel_kms"))), axis=1)
            return 0.7 * pd.Series(proba, index=df.index) + 0.3 * heur

        # Fallback: take the first n_features_in_ columns (only if they exist)
        if hasattr(model, "n_features_in_"):
            n = int(model.n_features_in_)
            cols = df.columns[:n]
            X = df[cols].copy().fillna(0.0)
            proba = model.predict_proba(X)[:, 1]
            heur = df.apply(lambda r: compute_probability(_safe_float(r.get("miss_km")),
                                                          _safe_float(r.get("vrel_kms"))), axis=1)
            return 0.7 * pd.Series(proba, index=df.index) + 0.3 * heur

    except Exception:
        # Any mismatch → heuristic only
        pass

    return df.apply(lambda r: compute_probability(_safe_float(r.get("miss_km")),
                                                  _safe_float(r.get("vrel_kms"))), axis=1)


# -------------------------------
# Balanced selection by risk buckets
# -------------------------------
def _balanced_pick(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Return up to `top_n` rows with a balanced mix of risks.
    Target mix: 2 Critical, 2 High, 1 Medium, 1 Low (for top_n >= 6).
    If some buckets are empty, backfill by overall probability.
    """
    # Sort once by probability desc
    df_sorted = df.sort_values("probability", ascending=False)

    # Bucketize
    buckets = {
        "Critical": df_sorted[df_sorted["probability"] >= 0.8],
        "High": df_sorted[(df_sorted["probability"] >= 0.6) & (df_sorted["probability"] < 0.8)],
        "Medium": df_sorted[(df_sorted["probability"] >= 0.3) & (df_sorted["probability"] < 0.6)],
        "Low": df_sorted[df_sorted["probability"] < 0.3],
    }

    # Target counts (scale proportionally for smaller/larger top_n)
    base_targets = {"Critical": 2, "High": 2, "Medium": 1, "Low": 1}
    base_total = sum(base_targets.values())
    scale = max(1.0, top_n / base_total)
    targets = {k: max(1, int(round(v * scale))) for k, v in base_targets.items()}

    # First pass: take up to target from each bucket
    picks = []
    picked_ids = set()
    for k in ["Critical", "High", "Medium", "Low"]:
        subset = buckets[k]
        take = subset.head(targets[k])
        for idx in take.index:
            if idx not in picked_ids:
                picks.append(idx)
                picked_ids.add(idx)
        if len(picks) >= top_n:
            break

    # Backfill from overall ranking if needed
    if len(picks) < top_n:
        for idx in df_sorted.index:
            if idx not in picked_ids:
                picks.append(idx)
                picked_ids.add(idx)
            if len(picks) >= top_n:
                break

    return df.loc[picks].sort_values("probability", ascending=False)


# -------------------------------
# Main: Predict Top Events
# -------------------------------
def predict_top_events(top_n: int = 6) -> Dict[str, object]:
    """
    Return a balanced top-N list (no 2-day cutoff).
    Requires CSV columns at least: i_name, j_name, tca, miss_km, vrel_kms
    """
    try:
        df = pd.read_csv(CSV_PATH)

        # Parse datetime (only for display / time-to-impact)
        df["EPOCH_dt"] = pd.to_datetime(df["tca"], errors="coerce", utc=True)

        # Compute probability (model if compatible, else heuristic)
        df["probability"] = _model_probability_block(df).clip(0.0, 0.98)

        # Build risk level text now (used for bucketizing and response)
        df["risk_level"] = df["probability"].apply(lambda p: classify_risk(float(p))[0])

        # Pick a balanced mix
        selected = _balanced_pick(df, top_n=top_n)

        results: List[Dict[str, str]] = []
        for _, row in selected.iterrows():
            prob = float(row["probability"])
            risk_level, maneuver = classify_risk(prob)

            sat_name = str(row.get("i_name", ""))
            debris_name = str(row.get("j_name", ""))

            results.append({
                "satellite": sat_name,
                "satellite_tle": fetch_tle(sat_name),
                "debris": debris_name,
                "debris_tle": fetch_tle(debris_name),
                "tca": row.get("tca"),
                "time_to_impact": time_to_impact(row.get("tca")),
                "miss_km": float(_safe_float(row.get("miss_km"))),
                "vrel_kms": float(_safe_float(row.get("vrel_kms"))),
                "probability": f"{prob*100:.1f}%",
                "risk_level": risk_level,
                "maneuver_suggestion": maneuver,
                "confidence": f"{prob*100:.1f}%"
            })

        return {"critical_events": results, "status": "ok"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
