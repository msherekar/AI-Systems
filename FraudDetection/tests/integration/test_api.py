# tests/integration/test_api.py
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json
import time
from threading import Thread
from src.api.service import FraudDetectionService
import requests

@pytest.fixture
def api_service(test_config):
    """Fixture to run API service for testing."""
    # Update config to use a different port for testing
    test_config['api']['port'] = 8787
    
    # Initialize service
    service = FraudDetectionService(test_config)
    
    # Start service in a separate thread
    thread = Thread(target=service.run)
    thread.daemon = True
    thread.start()
    
    # Wait for service to start
    time.sleep(2)
    
    yield service
    
    # No need to stop the thread as it's a daemon

def test_health_endpoint(api_service, test_config):
    """Test the health endpoint of the API."""
    # Make request to health endpoint
    response = requests.get(f"http://localhost:{test_config['api']['port']}/health")
    
    # Check response
    assert response.status_code == 200
    
    data = response.json()
    assert data['status'] == 'ok'
    assert data['model'] == test_config['api']['default_model']

def test_detect_fraud_post(api_service, test_config, sample_transaction_data):
    """Test the POST endpoint for fraud detection."""
    _, csv_path = sample_transaction_data
    
    # Create a temporary file with sample data
    with open(csv_path, 'rb') as f:
        # Make request to detect endpoint
        response = requests.post(
            f"http://localhost:{test_config['api']['port']}/detect",
            files={'input_transaction': f}
        )
    
    # Check response
    assert response.status_code == 200
    
    data = response.json()
    assert data['status'] == 'success'
    assert 'result' in data
    assert isinstance(data['result'], str)
    assert 'FRAUD' in data['result'] or 'LEGITIMATE' in data['result']

def test_detect_fraud_get(api_service, test_config, sample_transaction_data):
    """Test the GET endpoint for fraud detection."""
    _, csv_path = sample_transaction_data
    
    # Make request to infer endpoint
    response = requests.get(
        f"http://localhost:{test_config['api']['port']}/infer",
        params={'check_transaction': csv_path}
    )
    
    # Check response
    assert response.status_code == 200
    
    data = response.json()
    assert data['status'] == 'success'
    assert 'result' in data
    assert isinstance(data['result'], str)
    assert 'FRAUD' in data['result'] or 'LEGITIMATE' in data['result']