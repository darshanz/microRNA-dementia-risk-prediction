from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

from src.models.supervised_pca import DementiaRiskPredictor

app = FastAPI(title="Dementia Risk Prediction API")

model = DementiaRiskPredictor.load("models/ad_model.pkl")

class PredictionRequest(BaseModel):
    miRNA_expression: Dict[str, float]
    age: float
    sex: str
    apoe: float

class PredictionResponse(BaseModel):
    prediction: float
    risk_level: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict dementia risk from miRNA data"""
    try:
        # Convert to DataFrame
        X = pd.DataFrame([request.miRNA_expression])
        clinical = pd.DataFrame({
            'age': [request.age],
            'sex': [1 if request.sex == 'male' else 0],
            'apoe': [request.apoe]
        })
        
        # Make prediction
        proba = model.predict_proba(X, clinical)[0, 1]
        
        # Determine risk level
        if proba < 0.3:
            risk = "Low"
        elif proba < 0.7:
            risk = "Medium"
        else:
            risk = "High"
        
        return PredictionResponse(
            prediction=proba,
            risk_level=risk,
            confidence=proba
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}