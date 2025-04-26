# Test for API service

import unittest
from fastapi.testclient import TestClient
import os
import json
import pandas as pd
import tempfile
import pickle
from unittest.mock import patch, MagicMock
from src.api.service import app

class TestAPIService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Create a temporary directory for mock models
        self.temp_dir = tempfile.mkdtemp()
        self.original_model_dir = os.path.join(os.getcwd(), 'models')
        
    def tearDown(self):
        # Clean up
        pass
    
    def test_root_endpoint(self):
        # Test the root endpoint
        response = self.client.get("/")
        
        # Check status code
        self.assertEqual(response.status_code, 200)
        
        # Check response content
        self.assertEqual(response.json(), {"message": "Welcome to the TimeSeries Forecasting API!"})
    
    @patch('src.api.service.sarimax_total_model')
    def test_forecast_sarimax_total(self, mock_model):
        # Mock the forecast method to return a known value
        mock_model.forecast.return_value = [100, 110, 120]
        
        # Make request to the endpoint
        response = self.client.get("/forecast/sarimax/total?steps=3")
        
        # Check status code
        self.assertEqual(response.status_code, 200)
        
        # Check response content
        self.assertEqual(response.json(), {"forecast": [100, 110, 120]})
        
        # Verify that the model's forecast method was called with the right arguments
        mock_model.forecast.assert_called_once_with(steps=3)
    
    @patch('src.api.service.sarimax_fraud_model')
    def test_forecast_sarimax_fraud(self, mock_model):
        # Mock the forecast method to return a known value
        mock_model.forecast.return_value = [10, 11, 12]
        
        # Make request to the endpoint
        response = self.client.get("/forecast/sarimax/fraud?steps=3")
        
        # Check status code
        self.assertEqual(response.status_code, 200)
        
        # Check response content
        self.assertEqual(response.json(), {"forecast": [10, 11, 12]})
        
        # Verify that the model's forecast method was called with the right arguments
        mock_model.forecast.assert_called_once_with(steps=3)
    
    @patch('src.api.service.lstm_total_model')
    def test_forecast_lstm_total(self, mock_model):
        # Mock the LSTM model
        mock_model.is_defined = True
        
        # Make request to the endpoint
        response = self.client.get("/forecast/lstm/total?steps=3")
        
        # Check status code
        self.assertEqual(response.status_code, 200)
    
    @patch('src.api.service.lstm_fraud_model')
    def test_forecast_lstm_fraud(self, mock_model):
        # Mock the LSTM model
        mock_model.is_defined = True
        
        # Make request to the endpoint
        response = self.client.get("/forecast/lstm/fraud?steps=3")
        
        # Check status code
        self.assertEqual(response.status_code, 200)
    
    def test_forecast_sarimax_total_not_found(self):
        # Test error handling when model is not found
        with patch('src.api.service.sarimax_total_model', None):
            # Make request to the endpoint
            response = self.client.get("/forecast/sarimax/total?steps=3")
            
            # Check status code
            self.assertEqual(response.status_code, 404)
            
            # Check error message
            self.assertEqual(response.json(), {"detail": "SARIMAX Total model not found"})
    
    def test_forecast_upload_invalid_file(self):
        # Test error handling for invalid file upload
        
        # Create a text file (not CSV)
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            temp_file.write(b"This is not a CSV file")
            temp_file.flush()
            
            # Upload the file
            with open(temp_file.name, 'rb') as f:
                response = self.client.post(
                    "/forecast/upload",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            # Check status code
            self.assertEqual(response.status_code, 400)
            
            # Check error message
            self.assertEqual(response.json(), {"detail": "File must be a CSV"})
    
    def test_forecast_upload_valid_file(self):
        # Test file upload with a valid CSV
        
        # Create a simple CSV file
        df = pd.DataFrame({
            'trans_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'trans_amount': [100.0, 200.0, 300.0],
            'is_fraud': [0, 1, 0]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_file.flush()
            
            # Upload the file
            with open(temp_file.name, 'rb') as f:
                response = self.client.post(
                    "/forecast/upload",
                    files={"file": ("test.csv", f, "text/csv")}
                )
            
            # Check status code
            self.assertEqual(response.status_code, 200)
            
            # Check response content
            self.assertIn("message", response.json())

    def preprocess(self, df):
        df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
        df.dropna(subset=['trans_date'], inplace=True)  # Drop rows with invalid dates
        # Continue with the rest of your preprocessing logic

if __name__ == '__main__':
    unittest.main() 