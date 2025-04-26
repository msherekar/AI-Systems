#!/usr/bin/env python3
"""
Script for training the sentiment analysis model.
"""
import sys
import os
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.preprocess import normalization
from src.metrics.metrics import Metrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_path, text_column='text', label_column='sentiment'):
    """
    Load data from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file
        text_column (str): Name of the text column
        label_column (str): Name of the label column
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Ensure required columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in data")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in data")
    
    logger.info(f"Data loaded successfully: {len(df)} samples")
    return df

def preprocess_data(df, text_column='text'):
    """
    Preprocess the data.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        text_column (str): Name of the text column
        
    Returns:
        pandas.DataFrame: Preprocessed dataframe
    """
    logger.info("Preprocessing data")
    
    # Normalize text
    df['normalized_text'] = df[text_column].apply(normalization)
    
    # Remove rows with empty normalized text
    df_clean = df[df['normalized_text'].str.len() > 0].reset_index(drop=True)
    
    # Tokenize text
    df_clean['tokens'] = df_clean['normalized_text'].apply(word_tokenize)
    
    logger.info(f"Preprocessing complete: {len(df_clean)} samples retained")
    return df_clean

def train_word2vec(tokenized_texts, vector_size=100, window=5, min_count=1, epochs=10):
    """
    Train a Word2Vec model.
    
    Args:
        tokenized_texts (list): List of tokenized texts
        vector_size (int): Size of the embeddings
        window (int): Window size
        min_count (int): Minimum count of words
        epochs (int): Number of training epochs
        
    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model
    """
    logger.info(f"Training Word2Vec model: vector_size={vector_size}, window={window}, min_count={min_count}")
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs
    )
    logger.info("Word2Vec model training complete")
    return model

def create_embeddings(df, model_wv, max_length=100):
    """
    Create embeddings from tokenized texts.
    
    Args:
        df (pandas.DataFrame): Input dataframe with 'tokens' column
        model_wv (gensim.models.Word2Vec): Word2Vec model
        max_length (int): Maximum sequence length
        
    Returns:
        numpy.ndarray: Embeddings
    """
    logger.info("Creating embeddings")
    embeddings = []
    
    for tokens_list in df['tokens']:
        # Only include tokens that are in the vocabulary
        tokens_in_vocab = [token for token in tokens_list if token in model_wv.wv]
        if tokens_in_vocab:
            embeddings.append(np.array([model_wv.wv[token] for token in tokens_in_vocab]))
        else:
            # If no tokens are in vocabulary, use zeros
            embeddings.append(np.zeros((1, model_wv.vector_size)))
    
    # Apply padding or truncation and flatten the embeddings
    X = []
    for emb in embeddings:
        if len(emb) >= max_length:
            emb = emb[:max_length]
        else:
            # Pad with zeros
            pad_width = ((0, max_length - len(emb)), (0, 0))
            emb = np.pad(emb, pad_width, mode='constant')
        X.append(emb.flatten())
    
    X = np.array(X)
    logger.info(f"Embeddings created: {X.shape}")
    return X

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        n_estimators (int): Number of estimators in the random forest
        random_state (int): Random state for reproducibility
        
    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained model
    """
    logger.info(f"Training Random Forest model: n_estimators={n_estimators}")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model.
    
    Args:
        model: Trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model")
    predictions = model.predict(X_test)
    
    metrics = Metrics()
    evaluation = metrics.evaluate(y_test, predictions)
    
    logger.info(f"Model evaluation: Accuracy={evaluation['accuracy']:.4f}, F1={evaluation['f1']:.4f}")
    return evaluation

def save_model(model, model_path):
    """
    Save the model to disk.
    
    Args:
        model: Trained model
        model_path (str): Path to save the model
    """
    logger.info(f"Saving model to {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved successfully")

def save_word2vec(model, model_path):
    """
    Save the Word2Vec model to disk.
    
    Args:
        model (gensim.models.Word2Vec): Word2Vec model
        model_path (str): Path to save the model
    """
    logger.info(f"Saving Word2Vec model to {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info("Word2Vec model saved successfully")

def main(args):
    """
    Main function for training the model.
    
    Args:
        args: Command-line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    data_path = os.path.join(project_root, config['data']['raw_data_path'], args.data_file)
    df = load_data(data_path, text_column=config['data']['text_column'])
    
    # Preprocess data
    df_clean = preprocess_data(df, text_column=config['data']['text_column'])
    
    # Split data
    train_size = config['training']['train_size']
    test_size = config['training']['test_size'] / (1 - train_size)  # Adjust test_size for train_test_split
    random_state = config['training']['random_state']
    
    train_df, test_df = train_test_split(
        df_clean, 
        test_size=test_size, 
        random_state=random_state
    )
    
    logger.info(f"Data split: {len(train_df)} train samples, {len(test_df)} test samples")
    
    # Train Word2Vec model
    word2vec_config = config['model']['word2vec']
    word2vec_model = train_word2vec(
        train_df['tokens'],
        vector_size=word2vec_config['vector_size'],
        window=word2vec_config['window'],
        min_count=word2vec_config['min_count']
    )
    
    # Create embeddings
    X_train = create_embeddings(train_df, word2vec_model, max_length=config['data']['max_sequence_length'])
    X_test = create_embeddings(test_df, word2vec_model, max_length=config['data']['max_sequence_length'])
    
    y_train = train_df['sentiment'].values
    y_test = test_df['sentiment'].values
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluation = evaluate_model(model, X_test, y_test)
    
    # Save model and Word2Vec model
    model_path = os.path.join(project_root, config['model']['model_path'])
    word2vec_model_path = os.path.join(project_root, config['model']['word2vec']['model_path'])
    
    save_model(model, model_path)
    save_word2vec(word2vec_model, word2vec_model_path)
    
    # Display evaluation results
    print("\nModel Evaluation:")
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"F1 Score: {evaluation['f1']:.4f}")
    print("\nClassification Report:")
    print(evaluation['classification_report'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model")
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="enhanced_synthetic_reviews.csv",
        help="Name of the data file in the raw data directory"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.join(project_root, "config/config.yaml"),
        help="Path to the configuration file"
    )
    
    args = parser.parse_args()
    main(args) 