# tests/integration/test_pipeline.py
import pytest
import pandas as pd
import numpy as np
import os
from src.data.etl import ETLPipeline
from src.data.dataset import FraudDataset
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel

def test_etl_to_dataset(test_config, sample_transaction_data):
    """Test integration between ETL and Dataset components."""
    _, csv_path = sample_transaction_data
    
    # Initialize ETL pipeline
    etl = ETLPipeline(test_config)
    
    # Process data
    processed_path = etl.process_file(csv_path, 'test_integration.csv')
    
    # Initialize dataset handler
    dataset = FraudDataset(test_config)
    
    # Load and prepare data
    train_df, test_df = dataset.prepare_data(processed_path, balance=False)
    
    # Check if data preparation was successful
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert not train_df.empty
    assert not test_df.empty
    
    # Check if CV folds were created
    assert dataset.split_indices is not None
    assert len(dataset.split_indices) == test_config['dataset']['k_folds']
    
    # Get fold data
    X_train, y_train, X_val, y_val = dataset.get_fold_data(0)
    
    # Check if fold data is valid
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_val, pd.Series)
    
    # Clean up
    os.remove(processed_path)

def test_full_pipeline(test_config, sample_transaction_data):
    """Test the full pipeline from ETL to model training and prediction."""
    _, csv_path = sample_transaction_data
    
    # Initialize ETL pipeline
    etl = ETLPipeline(test_config)
    
    # Process data
    processed_path = etl.process_file(csv_path, 'test_full_pipeline.csv')
    
    # Initialize dataset handler
    dataset = FraudDataset(test_config)
    
    # Load and prepare data
    dataset.prepare_data(processed_path, balance=False)
    
    # Get fold data
    X_train, y_train, X_val, y_val = dataset.get_fold_data(0)
    
    # Initialize model
    model = RandomForestModel(test_config)
    
    # Train model
    model.train(X_train, y_train)
    
    # Make predictions
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    
    # Check if predictions are valid
    assert isinstance(train_predictions, np.ndarray)
    assert isinstance(val_predictions, np.ndarray)
    assert len(train_predictions) == len(X_train)
    assert len(val_predictions) == len(X_val)
    
    # Clean up
    os.remove(processed_path)