from fastapi import FastAPI
from app.model import predict_risk
from datetime import datetime, timedelta, timezone

app = FastAPI(title="Satellite Collision Risk API")

@app.get("/")
def root():
    return {"message": "🚀 Satellite Collision Risk API is running"}

@app.post("/predict_batch")
def predict_batch(events: list[dict]):
    """
    Takes a list of conjunction events (each dict = one row of features).
    Returns the top 4 most critical by probability.
    
    Example input:
    [
      {"satellite": "SAT-1", "debris": "DEB-1", "i_ecc":0.01, "j_ecc":0.02, "i_incl":55, "j_incl":56,
       "i_sma":6780, "j_sma":6795, "vrel_kms":10, "i_raan":120, "j_raan":121, "tca": "2025-08-20T12:00:00Z"},
      {...}, {...}, {...}
    ]
    """
    results = []
    for ev in events:
        try:
            result = predict_risk(
                ev["i_ecc"], ev["j_ecc"],
                ev["i_incl"], ev["j_incl"],
                ev["i_sma"], ev["j_sma"],
                ev["vrel_kms"],
                ev["i_raan"], ev["j_raan"]
            )

            prob_pct = f"{int(result['probability']*100)}%"
            confidence_pct = prob_pct

            # Handle TCA/time_to_impact
            if "tca" in ev and ev["tca"]:
                tca_time = datetime.fromisoformat(ev["tca"].replace("Z", "+00:00"))
            else:
                tca_time = datetime.now(timezone.utc) + timedelta(hours=12)

            delta = tca_time - datetime.now(timezone.utc)
            if delta.total_seconds() >= 0:
                time_to_impact = f"{delta.seconds//3600}h {(delta.seconds//60)%60}m"
            else:
                hours = abs(delta.seconds//3600)
                mins = abs((delta.seconds//60)%60)
                time_to_impact = f"-{hours}h {mins}m"

            results.append({
                "satellite": ev.get("satellite", "UNKNOWN"),
                "debris": ev.get("debris", "UNKNOWN"),
                "tca": tca_time.isoformat(),
                "time_to_impact": time_to_impact,
                "probability": prob_pct,
                "risk_level": result["risk_level"],
                "maneuver_suggestion": result["maneuver_suggestion"],
                "confidence": confidence_pct
            })
        except Exception as e:
            results.append({"error": str(e), "event": ev})

    # ✅ Sort by highest probability
    results_sorted = sorted(results, key=lambda x: int(x["probability"].replace("%", "")), reverse=True)

    # ✅ Return only top 4 critical cases
    return {"critical_events": results_sorted[:4]}
