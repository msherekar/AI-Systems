# tests/unit/test_models.py
import pytest
import numpy as np
import pandas as pd
import os
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel

def test_random_forest_init(test_config):
    """Test initialization of RandomForestModel."""
    # Initialize model
    model = RandomForestModel(test_config)
    
    # Check if model was initialized correctly
    assert model.n_estimators == test_config['models']['random_forest']['n_estimators']
    assert model.random_state == test_config['models']['random_forest']['random_state']
    assert model.use_pca == test_config['models']['random_forest']['use_pca']
    assert model.model is None

def test_random_forest_build_pipeline(test_config):
    """Test build_pipeline method of RandomForestModel."""
    # Initialize model
    model = RandomForestModel(test_config)
    
    # Build pipeline
    pipeline = model.build_pipeline()
    
    # Check if pipeline was built correctly
    assert 'scaler' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps
    
    # Check if PCA is included or not based on config
    if model.use_pca:
        assert 'pca' in pipeline.named_steps
    else:
        assert 'pca' not in pipeline.named_steps

def test_random_forest_train_predict(test_config, sample_processed_data):
    """Test train and predict methods of RandomForestModel."""
    df, _ = sample_processed_data
    
    # Split data into features and target
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    # Initialize model
    model = RandomForestModel(test_config)
    
    # Train model
    trained_model = model.train(X, y)
    
    # Check if model was trained correctly
    assert model.model is not None
    assert trained_model is model.model
    
    # Make predictions
    predictions = model.predict(X)
    
    # Check if predictions are valid
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({0, 1})

def test_random_forest_save_load(test_config, sample_processed_data, tmpdir):
    """Test save and load methods of RandomForestModel."""
    df, _ = sample_processed_data
    
    # Split data into features and target
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    # Initialize model
    model = RandomForestModel(test_config)
    
    # Set model directory to temporary directory
    model.model_dir = str(tmpdir)
    
    # Train model
    model.train(X, y)
    
    # Save model
    save_path = model.save()
    
    # Check if model was saved correctly
    assert os.path.exists(save_path)
    
    # Initialize new model
    new_model = RandomForestModel(test_config)
    new_model.model_dir = str(tmpdir)
    
    # Load model
    loaded_model = new_model.load()
    
    # Check if model was loaded correctly
    assert new_model.model is not None
    assert loaded_model is new_model.model
    
    # Make predictions with both models
    predictions1 = model.predict(X)
    predictions2 = new_model.predict(X)
    
    # Check if predictions match
    assert np.array_equal(predictions1, predictions2)