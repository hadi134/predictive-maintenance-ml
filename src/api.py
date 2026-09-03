"""FastAPI service for predictive-maintenance inference."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import FEATURES

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")
MODEL_PATH = Path("artifacts/model.joblib")


class SensorReading(BaseModel):
    temperature_c: float = Field(ge=0, le=200)
    vibration_mm_s: float = Field(ge=0, le=50)
    pressure_bar: float = Field(ge=0, le=100)
    rpm: float = Field(ge=0, le=10_000)
    operating_hours: float = Field(ge=0)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first with: python -m src.train")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(reading: SensorReading) -> dict[str, float | int | str]:
    try:
        model = load_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    row = pd.DataFrame([[getattr(reading, name) for name in FEATURES]], columns=FEATURES)
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "label": "failure risk" if prediction else "normal operation",
        "failure_probability": round(probability, 4),
    }
