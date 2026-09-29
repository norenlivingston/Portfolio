"""
Stage 4 — Model Serving
FastAPI inference endpoint for the trained regression pipeline.

Usage (from projects/ directory):
    python 03_mlops/serve.py

Endpoints:
    GET  /health   → service status + expected feature names
    POST /predict  → returns a regression prediction
    POST /explain  → returns per-feature SHAP contributions for a prediction
"""
import logging

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_registry import explain_one, get_config, get_features, predict_one

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Regression Pipeline API",
    description="Serves predictions and explanations from the trained regression pipeline.",
    version="1.0.0",
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: dict = Field(
        ...,
        description=(
            "One entry per feature the trained model requires. The feature-selection "
            "step in the pipeline picks a different set of columns on every training "
            "run, so the required names are not fixed - call GET /health first and "
            "send every name it returns."
        ),
        examples=[{"<name from GET /health>": 1.0, "...": "..."}],
    )


class PredictResponse(BaseModel):
    prediction: float
    model_used: str


class ExplainResponse(BaseModel):
    prediction: float
    base_value: float
    contributions: dict


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "expected_features": get_features()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    result = predict_one(req.features)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    log.info("Prediction served: %.4f", result["prediction"])
    return PredictResponse(prediction=result["prediction"], model_used=str(get_config()["mlops"]["model_path"]))


@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest):
    result = explain_one(req.features)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return ExplainResponse(**result)


if __name__ == "__main__":
    cfg = get_config()["mlops"]
    uvicorn.run(app, host=cfg["host"], port=cfg["port"])
