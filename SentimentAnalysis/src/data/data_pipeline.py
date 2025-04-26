# *********************** Code for establishing the data pipeline ************************

# This code demonstrates how to establish a data pipeline for processing text for sentiment analysis.
# The pipeline consists of the following steps: 1) loading the data, 2) preprocessing the data, 3) normalizing the data,
# 4) tokenizing the data, 5) embedding the data

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to path for relative imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.preprocess import normalization
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pipeline:
    """
    A class for processing text data for sentiment analysis.
    
    This pipeline handles loading data, preprocessing, normalizing, tokenizing, and embedding text.
    
    Attributes:
        new_data (pandas.DataFrame): The loaded data
        new_reviews (pandas.DataFrame): The review text data
        new_embeddings (list): List of embeddings for each review
        new_X (numpy.ndarray): The processed embeddings ready for model input
    """
    
    def __init__(self, csv_file_path, text_column='text', word2vec_model_path='models/model_wv.bin', max_length=100):
        """
        Initialize the Pipeline class and process the data.
        
        Args:
            csv_file_path (str): Path to the CSV file with text data
            text_column (str): Name of the column containing the text data
            word2vec_model_path (str): Path to the Word2Vec model file
            max_length (int): Maximum length of text sequences
        """
        self.max_length = max_length
        
        try:
            # Load data from the provided CSV file
            logger.info(f"Loading data from {csv_file_path}")
            try:
                self.new_data = pd.read_csv(csv_file_path)
            except Exception as e:
                logger.error(f"Error loading data from {csv_file_path}: {str(e)}")
                raise
                
            # Check if text column exists
            if text_column not in self.new_data.columns:
                raise ValueError(f"Text column '{text_column}' not found in the data")
                
            # Extract the text data
            self.new_reviews = self.new_data[[text_column]]
            logger.info(f"Data loaded successfully: {len(self.new_reviews)} reviews")
            
            # Normalize and tokenize the text
            logger.info("Normalizing and tokenizing text")
            self.new_reviews['normalized_text'] = self.new_reviews[text_column].apply(lambda x: normalization(x))
            self.new_reviews['tokens'] = self.new_reviews['normalized_text'].apply(lambda x: word_tokenize(x))
            
            # Load the Word2Vec model
            model_path = word2vec_model_path
            if not os.path.exists(model_path):
                model_path = os.path.join(project_root, model_path)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Word2Vec model not found at {model_path}")
                    
            logger.info(f"Loading Word2Vec model from {model_path}")
            try:
                model_wv = Word2Vec.load(model_path)
                logger.info("Word2Vec model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Word2Vec model: {str(e)}")
                raise
                
            # Generate embeddings for the new data
            logger.info("Generating embeddings")
            self.new_embeddings = []
            for tokens_list in self.new_reviews['tokens']:
                # Only include tokens that are in the vocabulary
                tokens_in_vocab = [token for token in tokens_list if token in model_wv.wv]
                if tokens_in_vocab:
                    self.new_embeddings.append(np.array([model_wv.wv[token] for token in tokens_in_vocab]))
                else:
                    # If no tokens are in vocabulary, use zeros
                    self.new_embeddings.append(np.zeros((1, model_wv.vector_size)))
            
            # Apply padding or truncation and flatten the embeddings
            logger.info("Padding and flattening embeddings")
            self.new_X = []
            for emb in self.new_embeddings:
                if len(emb) >= self.max_length:
                    emb = emb[:self.max_length]
                else:
                    # Pad with zeros
                    pad_width = ((0, self.max_length - len(emb)), (0, 0))
                    emb = np.pad(emb, pad_width, mode='constant')
                self.new_X.append(emb.flatten())
            self.new_X = np.array(self.new_X)
            logger.info(f"Embeddings processed successfully: {self.new_X.shape}")
            
        except Exception as e:
            logger.error(f"Error in pipeline initialization: {str(e)}")
            raise
    
    def get_processed_data(self):
        """
        Get the processed data.
        
        Returns:
            numpy.ndarray: The processed embeddings
        """
        return self.new_X

# Example usage:
if __name__ == '__main__':
    try:
        # initialize the pipeline
        test_csv_file_path = os.path.join(project_root, 'data/raw/test_data.csv')
        if os.path.exists(test_csv_file_path):
            pipeline = Pipeline(test_csv_file_path)
            processed_data = pipeline.new_X
            print(f"Processed data shape: {processed_data.shape}")
        else:
            print(f"Test file not found at {test_csv_file_path}")
    except Exception as e:
        print(f"Error in pipeline processing: {str(e)}")
