"""
Local FastAPI Server for GAT-LSTM Price Prediction
Run: python local_server.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gat_lstm_model import GATLSTMForecaster

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GAT-LSTM Local API", version="1.0.0")

# CORS - allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
models = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PredictionRequest(BaseModel):
    mandi: str
    crop: str
    date: str  # YYYY-MM-DD


class PredictionResponse(BaseModel):
    predictions: List[float]
    dates: List[str]
    mandi: str
    crop: str


def load_model(crop_type: str):
    """Load model for specific crop"""
    logger.info(f"Loading {crop_type} model...")
    
    base_path = f"models/{crop_type}"
    
    # Load metadata
    with open(f"{base_path}/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    # Create model
    model = GATLSTMForecaster(
        num_mandis=metadata['config']['num_mandis'],
        num_crops=metadata['config']['num_crops'],
        num_features=metadata['config']['num_features'],
        forecast_horizon=metadata['config']['forecast_horizon']
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(f"{base_path}/best_model.pth", map_location=device))
    model.eval()
    
    # Load data
    data = pd.read_csv(f"{base_path}/data.csv")
    data['Market_ID'] = data['State'] + " | " + data['District'] + " | " + data['Market']
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Add weather features if needed
    if 't_mean' not in data.columns and 't00' in data.columns:
        temp_cols = [f't{i:02d}' for i in range(24)]
        data['t_mean'] = data[temp_cols].mean(axis=1)
    
    if 'r_mean' not in data.columns and 'r00' in data.columns:
        rain_cols = [f'r{i:02d}' for i in range(24)]
        data['r_mean'] = data[rain_cols].mean(axis=1)
    
    # Add temporal features
    data['Month'] = data['Date'].dt.month
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['DayOfYear'] = data['Date'].dt.dayofyear
    
    models[crop_type] = {
        'model': model,
        'metadata': metadata,
        'data': data
    }
    
    logger.info(f"✓ {crop_type} model loaded")


@app.on_event("startup")
async def startup():
    """Load models on startup"""
    logger.info(f"Starting server on device: {device}")
    
    # Load wheat model
    if os.path.exists("models/wheat"):
        load_model("wheat")
    
    # Load tomato model
    if os.path.exists("models/tomato"):
        load_model("tomato")
    
    logger.info(f"✓ Server ready! Loaded models: {list(models.keys())}")


def get_historical_data(df, mandi_name, end_date, sequence_length=60):
    """Get historical data for prediction"""
    end_date = pd.to_datetime(end_date)
    start_date = end_date - timedelta(days=sequence_length - 1)
    
    mask = (
        (df['Market_ID'] == mandi_name) &
        (df['Date'] >= start_date) &
        (df['Date'] <= end_date)
    )
    
    historical = df[mask].sort_values('Date')
    
    if len(historical) < sequence_length:
        raise ValueError(
            f"Not enough data. Need {sequence_length} days, got {len(historical)}"
        )
    
    features = historical[['Imp_Price', 'Month', 'DayOfWeek', 
                          'DayOfYear', 't_mean', 'r_mean']].values
    
    return features[-sequence_length:]


def predict(crop_type: str, mandi: str, crop: str, date: str):
    """Make prediction"""
    if crop_type not in models:
        raise HTTPException(status_code=503, detail=f"{crop_type} model not loaded")
    
    model_data = models[crop_type]
    model = model_data['model']
    metadata = model_data['metadata']
    df = model_data['data']
    
    # Validate
    if mandi not in metadata['mandi_to_idx']:
        raise ValueError(f"Unknown mandi: {mandi}")
    
    # Get historical data
    features = get_historical_data(df, mandi, date, 
                                   metadata['config']['sequence_length'])
    
    # Normalize
    features_norm = metadata['feature_scaler'].transform(features)
    
    # Prepare tensors
    X = torch.FloatTensor(features_norm).unsqueeze(0).to(device)
    mandi_idx = torch.LongTensor([metadata['mandi_to_idx'][mandi]]).to(device)
    crop_idx = torch.LongTensor([metadata['crop_to_idx'][crop]]).to(device)
    edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
    edge_weights = torch.zeros(0).to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model(X, edge_index, mandi_idx, crop_idx, edge_weights)
    
    # Denormalize
    preds_np = predictions.cpu().numpy().squeeze()
    preds_denorm = metadata['price_scaler'].inverse_transform(
        preds_np.reshape(-1, 1)
    ).flatten()
    
    # Generate dates
    start_date = pd.to_datetime(date) + timedelta(days=1)
    forecast_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                     for i in range(len(preds_denorm))]
    
    return preds_denorm.tolist(), forecast_dates


@app.post("/predict/wheat", response_model=PredictionResponse)
async def predict_wheat(request: PredictionRequest):
    """Predict wheat prices"""
    try:
        predictions, dates = predict("wheat", request.mandi, request.crop, request.date)
        return PredictionResponse(
            predictions=predictions,
            dates=dates,
            mandi=request.mandi,
            crop=request.crop
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/tomato", response_model=PredictionResponse)
async def predict_tomato(request: PredictionRequest):
    """Predict tomato prices"""
    try:
        predictions, dates = predict("tomato", request.mandi, request.crop, request.date)
        return PredictionResponse(
            predictions=predictions,
            dates=dates,
            mandi=request.mandi,
            crop=request.crop
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "device": device,
        "models_loaded": list(models.keys())
    }


@app.get("/mandis/{crop_type}")
async def get_mandis(crop_type: str):
    """Get available mandis for crop"""
    if crop_type not in models:
        raise HTTPException(status_code=404, detail=f"{crop_type} model not found")
    
    mandis = list(models[crop_type]['metadata']['mandi_to_idx'].keys())
    return {"mandis": mandis, "count": len(mandis)}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting GAT-LSTM Local Server")
    print("="*60)
    print(f"Device: {device}")
    print("API will be available at: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
