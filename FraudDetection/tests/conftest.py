# tests/conftest.py
import pytest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from threading import Thread

@pytest.fixture
def test_config():
    """Fixture to provide test configuration."""
    return {
        'data': {
            'processed_dir': 'tests/data/processed',
            'raw_dir': 'tests/data/raw',
        },
        'preprocessing': {
            'night_start_hour': 22,
            'night_end_hour': 4,
        },
        'dataset': {
            'k_folds': 2,
            'random_state': 42,
            'balance_samples': 10,
            'test_size': 0.2
        },
        'models': {
            'random_forest': {
                'n_estimators': 100,
                'random_state': 42,
                'use_pca': False
            },
            'logistic_regression': {
                'class_weight': 'balanced',
                'use_pca': False
            }
        },
        'api': {
            'host': 'localhost',
            'port': 8787,
            'debug': False,
            'default_model': 'random_forest'
        }
    }

@pytest.fixture
def sample_transaction_data():
    """Fixture to generate sample transaction data for testing."""
    # Create directory for test data if it doesn't exist
    os.makedirs('tests/data/raw', exist_ok=True)
    os.makedirs('tests/data/processed', exist_ok=True)
    
    # Generate 100 fake transactions
    n_samples = 100
    
    # Create a base timestamp for consistent data
    base_time = datetime.now() - timedelta(days=30)
    
    # Generate transaction data
    data = {
        'unix_time': [int((base_time + timedelta(hours=i)).timestamp()) for i in range(n_samples)],
        'amt': np.random.uniform(10, 1000, n_samples),
        'cc_num': np.random.choice(range(1000, 2000), n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 10% fraud rate
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV for tests
    csv_path = 'tests/data/raw/test_transactions.csv'
    df.to_csv(csv_path, index=False)
    
    return df, csv_path

@pytest.fixture
def sample_processed_data(sample_transaction_data):
    """Fixture to generate sample processed data for testing."""
    raw_df, _ = sample_transaction_data
    
    # Add time-based features
    df = raw_df.copy()
    df['datetime'] = pd.to_datetime(df['unix_time'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 4)).astype(int)
    
    # Add calendar features
    df['is_holidays'] = ((df['datetime'].dt.month == 12) & (df['datetime'].dt.day >= 24)).astype(int)
    df['is_post_holidays'] = (df['datetime'].dt.month == 1).astype(int)
    df['is_summer'] = ((df['datetime'].dt.month >= 5) & (df['datetime'].dt.month <= 9)).astype(int)
    
    # Select features
    processed_df = df[['amt', 'cc_num', 'is_night', 'is_holidays', 'is_post_holidays', 'is_summer', 'is_fraud']]
    
    # Save to CSV for tests
    csv_path = 'tests/data/processed/test_processed.csv'
    processed_df.to_csv(csv_path, index=False)
    
    return processed_df, csv_path

@pytest.fixture
def api_service(test_config, sample_processed_data):
    """Fixture to run API service for testing."""
    # Update config to use a different port for testing
    test_config['api']['port'] = 8787
    
    # Create and train a model first
    df, _ = sample_processed_data
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    from src.models.random_forest import RandomForestModel
    model = RandomForestModel(test_config)
    model.train(X, y)
    
    # Initialize service with pre-trained model
    from src.api.service import FraudDetectionService
    service = FraudDetectionService(test_config)
    service.model = model  # Use the pre-trained model
    
    # Start service in a separate thread
    thread = Thread(target=service.run)
    thread.daemon = True
    thread.start()
    
    # Wait for service to start
    time.sleep(2)
    
    yield service