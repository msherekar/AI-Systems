# src/run_pipeline.py
import os
import pandas as pd
import numpy as np
import logging
from .data.etl import ETLPipeline
from .models.random_forest import RandomForestModel
from src.config import load_config

# Set up basic logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the complete fraud detection pipeline."""
    # Paths to your actual data files
    input_file = "/Users/mukulsherekar/Documents/Creating_AI_Enabled_Systems/FraudDetection/data/raw/train_dataset.csv"
    test_file = "/Users/mukulsherekar/Documents/Creating_AI_Enabled_Systems/FraudDetection/data/raw/val_dataset.csv"
    
    # Create a basic config
    config = {
        'data': {
            'processed_dir': 'data/processed',
            'raw_dir': 'data/raw',
        },
        'preprocessing': {
            'night_start_hour': 22,
            'night_end_hour': 4,
        },
        'models': {
            'random_forest': {
                'n_estimators': 100,
                'random_state': 42,
                'use_pca': False
            }
        }
    }
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    logger.info("Starting ETL process...")
    # Initialize ETL pipeline
    etl = ETLPipeline(config)
    
    # Process training data
    logger.info(f"Processing training data from {input_file}")
    processed_train_path = etl.process_file(input_file, 'processed_train.csv')
    logger.info(f"Processed training data saved to {processed_train_path}")
    
    # Process test data
    logger.info(f"Processing test data from {test_file}")
    processed_test_path = etl.process_file(test_file, 'processed_test.csv')
    logger.info(f"Processed test data saved to {processed_test_path}")
    
    # Load processed data
    logger.info("Loading processed data...")
    train_df = pd.read_csv(processed_train_path)
    test_df = pd.read_csv(processed_test_path)
    
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Prepare data for model
    X_train = train_df.drop(columns=['is_fraud'])
    y_train = train_df['is_fraud']
    X_test = test_df.drop(columns=['is_fraud'])
    y_test = test_df['is_fraud']
    
    # Initialize model
    logger.info("Initializing Random Forest model...")
    model = RandomForestModel(config)
    
    # Train model
    logger.info("Training model...")
    model.train(X_train, y_train)
    
    # Evaluate model
    logger.info("Evaluating model on test data...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"Test Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Save model
    logger.info("Saving model...")
    model_path = model.save()
    logger.info(f"Model saved to {model_path}")
    
    # Try making a prediction on a single transaction
    logger.info("Testing with a single transaction...")
    sample_transaction = X_test.iloc[[0]]
    prediction = model.predict(sample_transaction)
    result = "FRAUD" if prediction[0] == 1 else "LEGITIMATE"
    logger.info(f"Sample transaction prediction: {result}")
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()