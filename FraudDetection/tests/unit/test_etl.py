# tests/unit/test_etl.py
import pytest
import pandas as pd
import numpy as np
import os
from src.data.etl import ETLPipeline

def test_extract(test_config, sample_transaction_data):
    """Test the extract method of ETLPipeline."""
    _, csv_path = sample_transaction_data
    
    # Initialize ETL pipeline
    etl = ETLPipeline(test_config)
    
    # Extract data
    df = etl.extract(csv_path)
    
    # Check if extraction was successful
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'unix_time' in df.columns
    assert 'amt' in df.columns
    assert 'cc_num' in df.columns
    assert 'is_fraud' in df.columns

def test_add_time_features(test_config, sample_transaction_data):
    """Test the add_time_features method of ETLPipeline."""
    df, _ = sample_transaction_data
    
    # Initialize ETL pipeline
    etl = ETLPipeline(test_config)
    
    # Add time features
    result = etl.add_time_features(df)
    
    # Check if time features were added correctly
    assert 'datetime' in result.columns
    assert 'hour' in result.columns
    assert 'is_night' in result.columns
    
    # Check if is_night is correctly calculated
    night_hours = [22, 23, 0, 1, 2, 3]
    day_hours = [h for h in range(4, 22)]
    
    night_transactions = result[result['hour'].isin(night_hours)]
    day_transactions = result[result['hour'].isin(day_hours)]
    
    assert all(night_transactions['is_night'] == 1)
    assert all(day_transactions['is_night'] == 0)

def test_add_calendar_features(test_config, sample_transaction_data):
    """Test the add_calendar_features method of ETLPipeline."""
    df, _ = sample_transaction_data
    
    # Add datetime column first (as it's needed by add_calendar_features)
    df['datetime'] = pd.to_datetime(df['unix_time'], unit='s')
    
    # Initialize ETL pipeline
    etl = ETLPipeline(test_config)
    
    # Add calendar features
    result = etl.add_calendar_features(df)
    
    # Check if calendar features were added correctly
    assert 'is_holidays' in result.columns
    assert 'is_post_holidays' in result.columns
    assert 'is_summer' in result.columns
    
    # Check a few specific cases
    holiday_transactions = result[result['datetime'].dt.month == 12][result['datetime'].dt.day >= 24]
    post_holiday_transactions = result[result['datetime'].dt.month == 1]
    summer_transactions = result[result['datetime'].dt.month.isin([5, 6, 7, 8, 9])]
    
    assert all(holiday_transactions['is_holidays'] == 1)
    assert all(post_holiday_transactions['is_post_holidays'] == 1)
    assert all(summer_transactions['is_summer'] == 1)

def test_select_features(test_config, sample_processed_data):
    """Test the select_features method of ETLPipeline."""
    df, _ = sample_processed_data
    
    # Initialize ETL pipeline
    etl = ETLPipeline(test_config)
    
    # Add extra columns to test feature selection
    df['extra_column'] = 1
    
    # Select features
    result = etl.select_features(df)
    
    # Check if correct features were selected
    expected_columns = ['amt', 'cc_num', 'is_night', 'is_holidays', 'is_post_holidays', 'is_summer', 'is_fraud']
    assert set(result.columns) == set(expected_columns)
    assert 'extra_column' not in result.columns

def test_transform(test_config, sample_transaction_data):
    """Test the transform method of ETLPipeline."""
    df, _ = sample_transaction_data
    
    # Initialize ETL pipeline
    etl = ETLPipeline(test_config)
    
    # Transform data
    result = etl.transform(df)
    
    # Check if transformation was successful
    expected_columns = ['amt', 'cc_num', 'is_night', 'is_holidays', 'is_post_holidays', 'is_summer', 'is_fraud']
    assert set(result.columns) == set(expected_columns)
    assert len(result) == len(df)

def test_save(test_config, sample_processed_data):
    """Test the save method of ETLPipeline."""
    df, _ = sample_processed_data
    
    # Initialize ETL pipeline
    etl = ETLPipeline(test_config)
    
    # Save data
    output_path = etl.save(df, 'test_save.csv')
    
    # Check if file was saved correctly
    assert os.path.exists(output_path)
    
    # Load saved file and compare with original
    saved_df = pd.read_csv(output_path)
    
    # Check if saved data matches original
    assert set(saved_df.columns) == set(df.columns)
    assert len(saved_df) == len(df)
    
    # Clean up
    os.remove(output_path)