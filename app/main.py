from fastapi import FastAPI
from app.model import predict_risk
from datetime import datetime, timezone

app = FastAPI(title="🚀 Satellite Collision Risk API")

@app.get("/")
def root():
    return {"message": "Satellite Collision Risk API is running!"}

@app.post("/predict_batch")
def predict_batch(events: list[dict]):
    """
    Accepts a batch of events with orbital parameters + IDs
    Returns top 4 critical events.
    """
    results = []
    for ev in events:
        try:
            result = predict_risk(
                ev["i_ecc"], ev["j_ecc"],
                ev["i_incl"], ev["j_incl"],
                ev["i_sma"], ev["j_sma"],
                ev["vrel_kms"],
                sat_id=ev.get("satellite"),
                debris_id=ev.get("debris")
            )
            # If TCA not provided, fallback already handled
            if "tca" in ev:
                result["tca"] = ev["tca"]
                result["time_to_impact"] = result["time_to_impact"]
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "event": ev})

    # Sort by probability
    results_sorted = sorted(
        results,
        key=lambda x: int(x.get("probability", "0").replace("%", "")),
        reverse=True
    )
    return {"critical_events": results_sorted[:4]}
