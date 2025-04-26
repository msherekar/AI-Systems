# API service using FastAPI

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import torch
import pickle
import os
from pathlib import Path
import io

# Import project modules
from src.utils.config_manager import ConfigManager
from src.models.sarimax_total import SarimaxTotalModel
from src.models.sarimax_fraud import SarimaxFraudModel
from src.models.lstm_total import LSTMTotalModel
from src.models.lstm_fraud import LSTMFraudModel

app = FastAPI(title="TimeSeries Forecasting API",
              description="API for forecasting total and fraudulent transactions",
              version="1.0.0")

# Initialize config manager
config_manager = ConfigManager()

# Load models
def load_models():
    """
    Load the trained models for prediction.

    Returns:
        tuple: Tuple containing the trained models.
    """
    base_path = Path(__file__).parent.parent.parent / "models"
    
    # SARIMAX models
    sarimax_total_path = base_path / "sarimax_total.pkl"
    sarimax_fraud_path = base_path / "sarimax_fraud.pkl"
    
    # LSTM models
    lstm_total_path = base_path / "lstm_total.pt"
    lstm_fraud_path = base_path / "lstm_fraud.pt"
    
    # Load SARIMAX models
    if os.path.exists(sarimax_total_path):
        with open(sarimax_total_path, 'rb') as f:
            sarimax_total_model = pickle.load(f)
    else:
        sarimax_total_model = None
    
    if os.path.exists(sarimax_fraud_path):
        with open(sarimax_fraud_path, 'rb') as f:
            sarimax_fraud_model = pickle.load(f)
    else:
        sarimax_fraud_model = None
    
    # Load LSTM models
    if os.path.exists(lstm_total_path):
        lstm_total_params = config_manager.get_model_params('lstm', 'total')
        lstm_total_model = LSTMTotalModel(
            input_size=lstm_total_params.get('input_size', 1),
            hidden_size=lstm_total_params.get('hidden_size', 50),
            num_layers=lstm_total_params.get('num_layers', 1)
        )
        lstm_total_model.load_state_dict(torch.load(lstm_total_path))
        lstm_total_model.eval()
    else:
        lstm_total_model = None
    
    if os.path.exists(lstm_fraud_path):
        lstm_fraud_params = config_manager.get_model_params('lstm', 'fraud')
        lstm_fraud_model = LSTMFraudModel(
            input_size=lstm_fraud_params.get('input_size', 1),
            hidden_size=lstm_fraud_params.get('hidden_size', 50),
            num_layers=lstm_fraud_params.get('num_layers', 1)
        )
        lstm_fraud_model.load_state_dict(torch.load(lstm_fraud_path))
        lstm_fraud_model.eval()
    else:
        lstm_fraud_model = None
    
    return sarimax_total_model, sarimax_fraud_model, lstm_total_model, lstm_fraud_model

# Load models
sarimax_total_model, sarimax_fraud_model, lstm_total_model, lstm_fraud_model = load_models()

@app.get('/')
async def root():
    """Root endpoint."""
    return {'message': 'Welcome to the TimeSeries Forecasting API!'}

@app.get('/forecast/sarimax/total')
async def forecast_sarimax_total(steps: int = 12):
    """
    Forecast total transactions using SARIMAX model.
    
    Args:
        steps (int): Number of steps to forecast.
    
    Returns:
        dict: Forecasted transactions.
    """
    if sarimax_total_model is None:
        raise HTTPException(status_code=404, detail="SARIMAX Total model not found")
    
    forecast = sarimax_total_model.forecast(steps=steps)
    
    return {'forecast': forecast.tolist() if hasattr(forecast, 'tolist') else forecast}

@app.get('/forecast/sarimax/fraud')
async def forecast_sarimax_fraud(steps: int = 12):
    """
    Forecast fraudulent transactions using SARIMAX model.
    
    Args:
        steps (int): Number of steps to forecast.
    
    Returns:
        dict: Forecasted transactions.
    """
    if sarimax_fraud_model is None:
        raise HTTPException(status_code=404, detail="SARIMAX Fraud model not found")
    
    forecast = sarimax_fraud_model.forecast(steps=steps)
    
    return {'forecast': forecast.tolist() if hasattr(forecast, 'tolist') else forecast}

@app.get('/forecast/lstm/total')
async def forecast_lstm_total(steps: int = 12):
    """
    Forecast total transactions using LSTM model.
    
    Args:
        steps (int): Number of steps to forecast.
    
    Returns:
        dict: Forecasted transactions.
    """
    if lstm_total_model is None:
        raise HTTPException(status_code=404, detail="LSTM Total model not found")
    
    # This is a placeholder - actual implementation would depend on the LSTM model
    return {'message': 'LSTM Total model forecast endpoint. Actual implementation would depend on the model.'}

@app.get('/forecast/lstm/fraud')
async def forecast_lstm_fraud(steps: int = 12):
    """
    Forecast fraudulent transactions using LSTM model.
    
    Args:
        steps (int): Number of steps to forecast.
    
    Returns:
        dict: Forecasted transactions.
    """
    if lstm_fraud_model is None:
        raise HTTPException(status_code=404, detail="LSTM Fraud model not found")
    
    # This is a placeholder - actual implementation would depend on the LSTM model
    return {'message': 'LSTM Fraud model forecast endpoint. Actual implementation would depend on the model.'}

@app.post('/forecast/upload')
async def forecast_from_upload(file: UploadFile = File(...)):
    """
    Forecast transactions from uploaded data.
    
    Args:
        file (UploadFile): Uploaded CSV file.
    
    Returns:
        dict: Forecasted transactions.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # This is a placeholder - actual implementation would depend on the models
        return {'message': 'File uploaded successfully. Actual forecast implementation would depend on the models.'}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 