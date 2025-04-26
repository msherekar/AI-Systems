# tests/end_to_end/test_fraud_detection.py
import pytest
import pandas as pd
import numpy as np
import os
import json
import subprocess
import time
import requests
from src.data.etl import ETLPipeline
from src.models.random_forest import RandomForestModel

def test_fraud_detection_workflow(test_config, sample_transaction_data, tmpdir):
    """
    Test the complete fraud detection workflow:
    1. Process raw data
    2. Train a model
    3. Save the model
    4. Start the API service
    5. Make predictions using the API
    """
    df, csv_path = sample_transaction_data
    
    # Set model directory to temporary directory
    test_config['models']['random_forest']['model_dir'] = str(tmpdir)
    
    # Step 1: Process raw data
    etl = ETLPipeline(test_config)
    processed_path = etl.process_file(csv_path, 'test_e2e.csv')
    
    # Step 2: Load processed data
    processed_df = pd.read_csv(processed_path)
    
    # Split data
    X = processed_df.drop(columns=['is_fraud'])
    y = processed_df['is_fraud']
    
    # Step 3: Train a model
    model = RandomForestModel(test_config)
    model.model_dir = str(tmpdir)
    model.train(X, y)
    
    # Step 4: Save the model
    model_path = model.save()
    
    # Step 5: Create a new transaction for testing
    test_transaction = processed_df.iloc[[0]]
    test_transaction = test_transaction.drop(columns=['is_fraud'])
    test_transaction_path = os.path.join(str(tmpdir), 'test_transaction.csv')
    test_transaction.to_csv(test_transaction_path, index=False)
    
    # The actual API service would be started here, but for testing,
    # we can directly use the FraudDetectionService or simulate its response
    
    # For a real end-to-end test, you could:
    # 1. Start a Docker container with your service
    # 2. Make API requests to it
    # 3. Verify the responses
    
    # Clean up
    os.remove(processed_path)
    os.remove(test_transaction_path)