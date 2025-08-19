from fastapi import FastAPI
from app.model import predict_risk

app = FastAPI(title="Satellite Collision Risk API")

@app.get("/")
def root():
    return {"message": "🚀 Satellite Collision Risk API is running"}

@app.get("/predict")
def predict(
    i_ecc: float,
    j_ecc: float,
    i_incl: float,
    j_incl: float,
    vrel_kms: float
):
    """
    Predict satellite collision risk.
    Example:
    /predict?i_ecc=0.01&j_ecc=0.02&i_incl=55&j_incl=56&vrel_kms=10
    """
    try:
        result = predict_risk(i_ecc, j_ecc, i_incl, j_incl, vrel_kms)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
