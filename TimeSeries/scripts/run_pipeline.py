# Run the entire TimeSeries forecasting pipeline

import os, sys
import torch
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
sys.path.append('.')
# Import project modules
from src.data.preprocessor import DataPreprocessor
from src.models.sarimax_total import SarimaxTotalModel
from src.models.sarimax_fraud import SarimaxFraudModel
from src.models.lstm_total import LSTMTotalModel
from src.models.lstm_fraud import LSTMFraudModel
from src.training.trainer import ModelTrainer
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

# Set up logging
logger = setup_logger("pipeline")

def create_sequences(data, seq_length):
    """
    Create sequences for LSTM model training.
    
    Args:
        data (np.ndarray): Input data.
        seq_length (int): Sequence length.
    
    Returns:
        tuple: X and y tensors.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))

def run_pipeline(generate_data=True):
    """
    Run the entire TimeSeries forecasting pipeline.

    Args:
        generate_data (bool): Whether to generate synthetic data.

    Returns:
        None
    """
    try:
        logger.info("Starting the TimeSeries forecasting pipeline...")

        # Create the necessary directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Get the configuration
        config_manager = ConfigManager()
        data_paths = config_manager.get_data_paths()
        training_params = config_manager.get_training_params()

        # Load or generate data
        if generate_data:
            logger.info("Generating synthetic data...")
            from scripts.generate_synthetic_data import generate_synthetic_data
            df = generate_synthetic_data(output_path=data_paths.get('raw_data_path', 'data/raw/credit_card_data.csv'))
        else:
            logger.info("Loading data...")
            preprocessor = DataPreprocessor(data_paths.get('raw_data_path', 'data/raw/credit_card_data.csv'))
            df = preprocessor.load_data()

        # Preprocess the data
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(data_paths.get('raw_data_path', 'data/raw/credit_card_data.csv'))
        df = preprocessor.preprocess(df)

        # Save processed data
        processed_data_path = data_paths.get('processed_data_path', 'data/processed/processed_data.csv')
        df.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data saved to {processed_data_path}")

        # Group data by day
        daily_transactions = df.groupby('trans_date').size()
        daily_transactions = daily_transactions.rename('total_transactions')
        daily_fraud = df[df['is_fraud'] == 1].groupby('trans_date').size()
        daily_fraud = daily_fraud.rename('fraud_transactions')

        # Convert to time series
        daily_transactions.index = pd.to_datetime(daily_transactions.index)
        daily_fraud.index = pd.to_datetime(daily_fraud.index)

        # Group by month for SARIMAX models
        monthly_transactions = daily_transactions.groupby(pd.Grouper(freq='MS')).sum()
        monthly_fraud = daily_fraud.groupby(pd.Grouper(freq='MS')).sum()

        # Train-test split
        train_test_split = training_params.get('train_test_split', 0.8)
        
        # For SARIMAX
        cutoff_total_monthly = int(len(monthly_transactions) * train_test_split)
        cutoff_fraud_monthly = int(len(monthly_fraud) * train_test_split)

        # Split data for SARIMAX
        train_total_monthly = monthly_transactions.iloc[:cutoff_total_monthly]
        test_total_monthly = monthly_transactions.iloc[cutoff_total_monthly:]
        train_fraud_monthly = monthly_fraud.iloc[:cutoff_fraud_monthly]
        test_fraud_monthly = monthly_fraud.iloc[cutoff_fraud_monthly:]

        # For LSTM
        cutoff_total_daily = int(len(daily_transactions) * train_test_split)
        cutoff_fraud_daily = int(len(daily_fraud) * train_test_split)

        # Split data for LSTM
        train_total_daily = daily_transactions.iloc[:cutoff_total_daily]
        test_total_daily = daily_transactions.iloc[cutoff_total_daily:]
        train_fraud_daily = daily_fraud.iloc[:cutoff_fraud_daily]
        test_fraud_daily = daily_fraud.iloc[cutoff_fraud_daily:]

        # ----------------------------------------------------------------------
        # Train SARIMAX models
        # ----------------------------------------------------------------------
        logger.info("Training SARIMAX models...")
        sarimax_total_params = config_manager.get_model_params('sarimax', 'total')
        sarimax_fraud_params = config_manager.get_model_params('sarimax', 'fraud')

        try:
            # Train SARIMAX total model
            sarimax_total_model = SarimaxTotalModel(
                order=sarimax_total_params.get('order', (1, 1, 1)),
                seasonal_order=sarimax_total_params.get('seasonal_order', (1, 1, 1, 12))
            )
            sarimax_total_model.fit(train_total_monthly)

            # Train SARIMAX fraud model
            sarimax_fraud_model = SarimaxFraudModel(
                order=sarimax_fraud_params.get('order', (1, 1, 1)),
                seasonal_order=sarimax_fraud_params.get('seasonal_order', (1, 1, 1, 12))
            )
            sarimax_fraud_model.fit(train_fraud_monthly)

            # Save SARIMAX models
            with open('models/sarimax_total.pkl', 'wb') as f:
                pickle.dump(sarimax_total_model, f)
            with open('models/sarimax_fraud.pkl', 'wb') as f:
                pickle.dump(sarimax_fraud_model, f)

            logger.info("SARIMAX models trained and saved.")

            # Evaluate SARIMAX models
            total_forecast = sarimax_total_model.forecast(steps=len(test_total_monthly))
            fraud_forecast = sarimax_fraud_model.forecast(steps=len(test_fraud_monthly))

            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            total_mse = mean_squared_error(test_total_monthly, total_forecast)
            total_mae = mean_absolute_error(test_total_monthly, total_forecast)
            
            fraud_mse = mean_squared_error(test_fraud_monthly, fraud_forecast)
            fraud_mae = mean_absolute_error(test_fraud_monthly, fraud_forecast)
            
            logger.info(f"SARIMAX Total Model - MSE: {total_mse:.4f}, MAE: {total_mae:.4f}")
            logger.info(f"SARIMAX Fraud Model - MSE: {fraud_mse:.4f}, MAE: {fraud_mae:.4f}")

        except Exception as e:
            logger.error(f"Error training SARIMAX models: {str(e)}")
            logger.warning("Continuing with LSTM models...")

        # ----------------------------------------------------------------------
        # Train LSTM models
        # ----------------------------------------------------------------------
        logger.info("Preparing data for LSTM models...")
        
        try:
            # Get LSTM parameters from config
            lstm_total_params = config_manager.get_model_params('lstm', 'total')
            lstm_fraud_params = config_manager.get_model_params('lstm', 'fraud')
            
            # Sequence length for LSTM models
            seq_length = 7  # Use a week of data to predict the next day
            
            # Prepare data for total transactions LSTM
            
            # Scale the data
            scaler_total = MinMaxScaler()
            train_total_values = train_total_daily.values.reshape(-1, 1)
            train_total_scaled = scaler_total.fit_transform(train_total_values)
            
            # Create sequences
            X_train_total, y_train_total = create_sequences(train_total_scaled, seq_length)
            
            # Prepare test data for evaluation
            test_total_values = test_total_daily.values.reshape(-1, 1)
            test_total_scaled = scaler_total.transform(test_total_values)
            X_test_total, y_test_total = create_sequences(test_total_scaled, seq_length)
            
            # Create DataLoader
            train_dataset_total = TensorDataset(X_train_total, y_train_total)
            train_loader_total = DataLoader(
                train_dataset_total, 
                batch_size=lstm_total_params.get('batch_size', 32),
                shuffle=True
            )
            
            # Prepare data for fraud transactions LSTM
            
            # Scale the data
            scaler_fraud = MinMaxScaler()
            train_fraud_values = train_fraud_daily.values.reshape(-1, 1)
            train_fraud_scaled = scaler_fraud.fit_transform(train_fraud_values)
            
            # Create sequences
            X_train_fraud, y_train_fraud = create_sequences(train_fraud_scaled, seq_length)
            
            # Prepare test data for evaluation
            test_fraud_values = test_fraud_daily.values.reshape(-1, 1)
            test_fraud_scaled = scaler_fraud.transform(test_fraud_values)
            X_test_fraud, y_test_fraud = create_sequences(test_fraud_scaled, seq_length)
            
            # Create DataLoader
            train_dataset_fraud = TensorDataset(X_train_fraud, y_train_fraud)
            train_loader_fraud = DataLoader(
                train_dataset_fraud, 
                batch_size=lstm_fraud_params.get('batch_size', 32),
                shuffle=True
            )
            
            # Train LSTM total model
            logger.info("Training LSTM total model...")
            lstm_total_model = LSTMTotalModel(
                input_size=lstm_total_params.get('input_size', 1),
                hidden_size=lstm_total_params.get('hidden_size', 50),
                num_layers=lstm_total_params.get('num_layers', 1)
            )
            
            # Define loss function and optimizer
            criterion_total = torch.nn.MSELoss()
            optimizer_total = torch.optim.Adam(lstm_total_model.parameters(), lr=0.001)
            
            # Create trainer for total model
            trainer_total = ModelTrainer(lstm_total_model, train_loader_total, criterion_total, optimizer_total)
            trainer_total.train(lstm_total_params.get('epochs', 30))
            
            # Save the total model
            torch.save(lstm_total_model.state_dict(), 'models/lstm_total.pt')
            
            # Train LSTM fraud model
            logger.info("Training LSTM fraud model...")
            lstm_fraud_model = LSTMFraudModel(
                input_size=lstm_fraud_params.get('input_size', 1),
                hidden_size=lstm_fraud_params.get('hidden_size', 50),
                num_layers=lstm_fraud_params.get('num_layers', 1)
            )
            
            # Define loss function and optimizer
            criterion_fraud = torch.nn.MSELoss()
            optimizer_fraud = torch.optim.Adam(lstm_fraud_model.parameters(), lr=0.001)
            
            # Create trainer for fraud model
            trainer_fraud = ModelTrainer(lstm_fraud_model, train_loader_fraud, criterion_fraud, optimizer_fraud)
            trainer_fraud.train(lstm_fraud_params.get('epochs', 30))
            
            # Save the fraud model
            torch.save(lstm_fraud_model.state_dict(), 'models/lstm_fraud.pt')
            
            # Also save the scalers for future prediction
            with open('models/scaler_total.pkl', 'wb') as f:
                pickle.dump(scaler_total, f)
            with open('models/scaler_fraud.pkl', 'wb') as f:
                pickle.dump(scaler_fraud, f)
            
            logger.info("LSTM models trained and saved.")
            
            # Evaluate LSTM models
            lstm_total_model.eval()
            with torch.no_grad():
                predictions_total = lstm_total_model(X_test_total)
                loss_total = criterion_total(predictions_total, y_test_total)
                
                # Inverse transform predictions
                predictions_total_np = predictions_total.numpy().reshape(-1, 1)
                y_test_total_np = y_test_total.numpy().reshape(-1, 1)
                
                predictions_total_inv = scaler_total.inverse_transform(predictions_total_np)
                y_test_total_inv = scaler_total.inverse_transform(y_test_total_np)
                
                total_mse = mean_squared_error(y_test_total_inv, predictions_total_inv)
                total_mae = mean_absolute_error(y_test_total_inv, predictions_total_inv)
                
                logger.info(f"LSTM Total Model - MSE: {total_mse:.4f}, MAE: {total_mae:.4f}, Loss: {loss_total.item():.4f}")
            
            lstm_fraud_model.eval()
            with torch.no_grad():
                predictions_fraud = lstm_fraud_model(X_test_fraud)
                loss_fraud = criterion_fraud(predictions_fraud, y_test_fraud)
                
                # Inverse transform predictions
                predictions_fraud_np = predictions_fraud.numpy().reshape(-1, 1)
                y_test_fraud_np = y_test_fraud.numpy().reshape(-1, 1)
                
                predictions_fraud_inv = scaler_fraud.inverse_transform(predictions_fraud_np)
                y_test_fraud_inv = scaler_fraud.inverse_transform(y_test_fraud_np)
                
                fraud_mse = mean_squared_error(y_test_fraud_inv, predictions_fraud_inv)
                fraud_mae = mean_absolute_error(y_test_fraud_inv, predictions_fraud_inv)
                
                logger.info(f"LSTM Fraud Model - MSE: {fraud_mse:.4f}, MAE: {fraud_mae:.4f}, Loss: {loss_fraud.item():.4f}")
                
            # Save evaluation metrics
            evaluation_metrics = {
                'sarimax_total': {'mse': total_mse, 'mae': total_mae},
                'sarimax_fraud': {'mse': fraud_mse, 'mae': fraud_mae},
                'lstm_total': {'mse': total_mse, 'mae': total_mae},
                'lstm_fraud': {'mse': fraud_mse, 'mae': fraud_mae}
            }
            
            with open('models/evaluation_metrics.pkl', 'wb') as f:
                pickle.dump(evaluation_metrics, f)
            
        except Exception as e:
            logger.error(f"Error training LSTM models: {str(e)}")
        
        logger.info("Pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the TimeSeries forecasting pipeline.')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic data.')
    parser.add_argument('--no-generate-data', action='store_false', dest='generate_data',
                        help='Do not generate synthetic data, use existing data.')
    args = parser.parse_args()

    run_pipeline(generate_data=args.generate_data) 