# Test for SARIMAX models

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from src.models.sarimax_total import SarimaxTotalModel
from src.models.sarimax_fraud import SarimaxFraudModel

class TestSarimaxModels(unittest.TestCase):
    def setUp(self):
        # Suppress warnings during tests
        warnings.filterwarnings("ignore")
        
        # Create a simple time series data for testing
        dates = pd.date_range(start='2023-01-01', periods=30, freq='MS')
        
        # Create a simple time series with a seasonal pattern
        values = np.sin(np.arange(0, 30) * (2 * np.pi / 12)) * 10 + 100  # Seasonal pattern with mean=100
        values += np.random.normal(0, 1, 30)  # Add some random noise
        
        # For fraud, create a similar but smaller series
        fraud_values = values * 0.1  # 10% of total are fraud
        
        # Create the series for total transactions
        self.total_series = pd.Series(values, index=dates)
        
        # Create the series for fraud transactions
        self.fraud_series = pd.Series(fraud_values, index=dates)
        
        # Initialize models
        self.total_model = SarimaxTotalModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        self.fraud_model = SarimaxFraudModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    
    def test_fit_total_model(self):
        # Fit the total model
        self.total_model.fit(self.total_series)
        
        # Check if the model is fitted
        self.assertIsNotNone(self.total_model.model)
    
    def test_fit_fraud_model(self):
        # Fit the fraud model
        self.fraud_model.fit(self.fraud_series)
        
        # Check if the model is fitted
        self.assertIsNotNone(self.fraud_model.model)
    
    def test_forecast_total_model(self):
        # Fit the model
        self.total_model.fit(self.total_series)
        
        # Forecast 5 steps ahead
        forecast = self.total_model.forecast(steps=5)
        
        # Check if the forecast has the right shape
        self.assertEqual(len(forecast), 5)
        
        # Check if the forecast values are reasonable (e.g., not NaN)
        self.assertFalse(np.isnan(forecast).any())
    
    def test_forecast_fraud_model(self):
        # Fit the model
        self.fraud_model.fit(self.fraud_series)
        
        # Forecast 5 steps ahead
        forecast = self.fraud_model.forecast(steps=5)
        
        # Check if the forecast has the right shape
        self.assertEqual(len(forecast), 5)
        
        # Check if the forecast values are reasonable (e.g., not NaN)
        self.assertFalse(np.isnan(forecast).any())
    
    def test_forecast_values_are_positive(self):
        # Since transactions can't be negative, check if forecasts are positive
        
        # Fit the total model
        self.total_model.fit(self.total_series)
        
        # Forecast 5 steps ahead
        forecast_total = self.total_model.forecast(steps=5)
        
        # Check if forecast values are positive
        self.assertTrue((forecast_total >= 0).all())
        
        # Fit the fraud model
        self.fraud_model.fit(self.fraud_series)
        
        # Forecast 5 steps ahead
        forecast_fraud = self.fraud_model.forecast(steps=5)
        
        # Check if forecast values are positive
        self.assertTrue((forecast_fraud >= 0).all())
    
    def test_model_with_missing_data(self):
        # Create a series with missing data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='MS')
        values = np.sin(np.arange(0, 30) * (2 * np.pi / 12)) * 10 + 100
        
        # Set some values to NaN
        values[5] = np.nan
        values[10] = np.nan
        
        # Create a series with NaN values
        series_with_nan = pd.Series(values, index=dates)
        
        try:
            # Fit the model with missing data
            self.total_model.fit(series_with_nan)
            
            # If it gets here, the model can handle missing data
            # Forecast 5 steps ahead
            forecast = self.total_model.forecast(steps=5)
            
            # Check if the forecast has the right shape
            self.assertEqual(len(forecast), 5)
        except Exception as e:
            # If the model can't handle missing data, this will catch the error
            self.fail(f"Model failed to handle missing data: {str(e)}")

if __name__ == '__main__':
    unittest.main() 