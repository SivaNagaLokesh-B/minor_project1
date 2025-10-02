# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
from typing import List, Dict, Optional
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for equipment failure prediction",
    version="1.0.0"
)

logger = logging.getLogger(__name__)

# Request/Response models
class EquipmentData(BaseModel):
    vibration: float
    temperature: float
    pressure: float
    current: float
    rpm: float
    tool_wear: float

class BatchEquipmentData(BaseModel):
    equipment_id: str
    vibration: float
    temperature: float
    pressure: float
    current: float
    rpm: float
    tool_wear: float

class PredictionResponse(BaseModel):
    equipment_id: str
    predicted_rul_hours: float
    failure_probability: float
    confidence: float
    maintenance_recommendation: str
    predicted_failure_date: str
    timestamp: str
    data_quality_warnings: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    expected_ranges_loaded: bool

# Global variables for expected ranges
EXPECTED_RANGES = {}

def load_expected_ranges():
    """Load expected input ranges from file"""
    global EXPECTED_RANGES
    try:
        ranges_file = "models/expected_ranges.json"
        if os.path.exists(ranges_file):
            with open(ranges_file, 'r') as f:
                EXPECTED_RANGES = json.load(f)
            logger.info("✅ Expected ranges loaded successfully")
            return True
        else:
            logger.warning("Expected ranges file not found: %s", ranges_file)
            return False
    except Exception as e:
        logger.error("Failed to load expected ranges: %s", e)
        return False

def validate_input_data(data: Dict) -> List[str]:
    """Validate input data against expected ranges"""
    warnings = []
    
    for field, value in data.items():
        if field in EXPECTED_RANGES:
            expected_min = EXPECTED_RANGES[field]['min']
            expected_max = EXPECTED_RANGES[field]['max']
            
            if value < expected_min or value > expected_max:
                warnings.append(
                    f"{field}={value} outside expected range [{expected_min:.2f}, {expected_max:.2f}]"
                )
    
    return warnings

# Load model and expected ranges at startup
@app.on_event("startup")
async def startup_event():
    try:
        app.state.pipeline = joblib.load("models/production_pipeline.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        app.state.pipeline = None
    
    # Load expected ranges for input validation
    app.state.expected_ranges_loaded = load_expected_ranges()

@app.get("/")
async def root():
    return {"message": "Predictive Maintenance API", "status": "healthy"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    scaler_loaded = os.path.exists("models/scalers/feature_scaler.pkl")
    return HealthResponse(
        status="healthy" if app.state.pipeline else "unhealthy",
        model_loaded=app.state.pipeline is not None,
        scaler_loaded=scaler_loaded,
        expected_ranges_loaded=getattr(app.state, 'expected_ranges_loaded', False)
    )

@app.get("/expected_ranges")
async def get_expected_ranges():
    """Return expected input ranges for clients"""
    if not EXPECTED_RANGES:
        raise HTTPException(status_code=404, detail="Expected ranges not loaded")
    
    return EXPECTED_RANGES

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: EquipmentData, equipment_id: str = "unknown"):
    """Predict equipment failure probability and RUL"""
    if not app.state.pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = data.dict()
        input_df = pd.DataFrame([input_data])
        input_df['equipment_id'] = equipment_id
        
        # Validate input data
        data_quality_warnings = []
        if EXPECTED_RANGES:
            data_quality_warnings = validate_input_data(input_data)
            if data_quality_warnings:
                logger.warning("Input validation warnings for %s: %s", equipment_id, data_quality_warnings)
        
        # Make prediction
        result = app.state.pipeline.predict(input_df)
        
        # Check for errors
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        logger.info("Prediction for %s: RUL=%.1fh, failure_prob=%.2f", 
                   equipment_id, 
                   result.get('predicted_rul_hours', 0),
                   result.get('failure_probability', 0))
        
        return PredictionResponse(
            equipment_id=equipment_id,
            predicted_rul_hours=result.get('predicted_rul_hours', 0),
            failure_probability=result.get('failure_probability', 0),
            confidence=result.get('confidence', 0),
            maintenance_recommendation=result.get('maintenance_recommendation', 'Unknown'),
            predicted_failure_date=result.get('predicted_failure_date', ''),
            timestamp=result.get('timestamp', datetime.now().isoformat()),
            data_quality_warnings=data_quality_warnings if data_quality_warnings else None
        )
        
    except Exception as e:
        logger.error("Prediction error for %s: %s", equipment_id, e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(data: List[BatchEquipmentData]):
    """Batch prediction for multiple equipment"""
    if not app.state.pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        all_warnings = []
        
        for item in data:
            # Extract equipment data (excluding equipment_id)
            equipment_data = item.dict()
            eq_id = equipment_data.pop('equipment_id')
            
            input_df = pd.DataFrame([equipment_data])
            input_df['equipment_id'] = eq_id
            
            # Validate input data
            data_quality_warnings = []
            if EXPECTED_RANGES:
                data_quality_warnings = validate_input_data(equipment_data)
                if data_quality_warnings:
                    logger.warning("Input validation warnings for %s: %s", eq_id, data_quality_warnings)
                    all_warnings.extend([f"{eq_id}: {warn}" for warn in data_quality_warnings])
            
            result = app.state.pipeline.predict(input_df)
            result['equipment_id'] = eq_id
            result['data_quality_warnings'] = data_quality_warnings if data_quality_warnings else None
            results.append(result)
        
        response = {"predictions": results}
        if all_warnings:
            response["batch_warnings"] = all_warnings
            
        return response
        
    except Exception as e:
        logger.error("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")