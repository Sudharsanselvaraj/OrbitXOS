from fastapi import FastAPI
from app.model import predict_top_events

app = FastAPI(title="🚀 Satellite Collision Risk API")


@app.get("/")
def root():
    return {"message": "API is running", "endpoints": ["/predict"]}


@app.get("/predict")
def predict():
    """Return top 4 most critical conjunction events."""
    return predict_top_events(top_n=4)
