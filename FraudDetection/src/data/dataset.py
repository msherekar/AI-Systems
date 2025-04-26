# src/data/dataset.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split
from typing import Dict, Any, Tuple, List, Optional
import os
import logging

logger = logging.getLogger(__name__)

class FraudDataset:
    """Dataset handler for fraud detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset handler.
        
        Args:
            config: Dictionary containing dataset configuration
        """
        self.config = config
        self.k_folds = config.get('dataset', {}).get('k_folds', 5)
        self.random_state = config.get('dataset', {}).get('random_state', 42)
        self.balance_samples = config.get('dataset', {}).get('balance_samples', 1000)
        self.test_size = config.get('dataset', {}).get('test_size', 0.2)
        self.processed_dir = config.get('data', {}).get('processed_dir', 'data/processed')
        
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.split_indices = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading dataset from {filepath}")
        
        try:
            self.data = pd.read_csv(filepath)
            logger.info(f"Loaded dataset with shape {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading dataset from {filepath}: {e}")
            raise
    
    def balance_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Balance the dataset by oversampling the minority class.
        
        Args:
            df: Input DataFrame. If None, use self.data.
            
        Returns:
            Balanced DataFrame
        """
        if df is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            df = self.data
            
        logger.info(f"Balancing dataset with shape {df.shape}")
        
        # Separate fraud and non-fraud transactions
        fraud_df = df[df['is_fraud'] == 1]
        non_fraud_df = df[df['is_fraud'] == 0]
        
        logger.info(f"Found {len(fraud_df)} fraud and {len(non_fraud_df)} non-fraud transactions")
        
        # Sample from each class
        if len(fraud_df) < self.balance_samples:
            logger.warning(f"Not enough fraud samples. Using all available {len(fraud_df)} samples.")
            sampled_fraud_df = fraud_df
        else:
            sampled_fraud_df = fraud_df.sample(n=self.balance_samples, random_state=self.random_state)
            
        sampled_non_fraud_df = non_fraud_df.sample(n=self.balance_samples, random_state=self.random_state)
        
        # Combine and shuffle
        balanced_df = pd.concat([sampled_fraud_df, sampled_non_fraud_df])
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        logger.info(f"Balanced dataset to shape {balanced_df.shape}")
        return balanced_df
    
    def train_test_split(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.
        
        Args:
            df: Input DataFrame. If None, use self.data.
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if df is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            df = self.data
            
        logger.info(f"Splitting dataset with shape {df.shape} into train and test sets")
        
        # Use GroupKFold to ensure all transactions from the same credit card stay together
        groups = df['cc_num']
        
        # Generate train-test split indices
        gkf = GroupKFold(n_splits=int(1/self.test_size))
        train_idx, test_idx = next(gkf.split(df, df['is_fraud'], groups))
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        logger.info(f"Split dataset into train ({train_df.shape}) and test ({test_df.shape}) sets")
        
        # Store the training and test data
        self.X_train = train_df.drop(columns=['is_fraud'])
        self.y_train = train_df['is_fraud']
        self.X_test = test_df.drop(columns=['is_fraud'])
        self.y_test = test_df['is_fraud']
        
        return train_df, test_df
    
    def create_cv_folds(self, df: Optional[pd.DataFrame] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create k-fold cross-validation indices.
        
        Args:
            df: Input DataFrame. If None, use self.data.
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if df is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            df = self.data
            
        logger.info(f"Creating {self.k_folds}-fold CV for dataset with shape {df.shape}")
        
        # Use GroupKFold to ensure all transactions from the same credit card stay together
        groups = df['cc_num']
        
        # Generate k-fold indices
        gkf = GroupKFold(n_splits=self.k_folds)
        self.split_indices = list(gkf.split(df, df['is_fraud'], groups))
        
        logger.info(f"Created {len(self.split_indices)} CV folds")
        return self.split_indices
    
    def get_fold_data(self, fold: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Get training and validation data for a specific fold.
        
        Args:
            fold: Fold index (0-based)
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        if self.split_indices is None:
            raise ValueError("No CV folds created. Call create_cv_folds() first.")
            
        if fold < 0 or fold >= len(self.split_indices):
            raise ValueError(f"Fold index {fold} out of range (0-{len(self.split_indices)-1})")
            
        train_idx, val_idx = self.split_indices[fold]
        
        X_train = self.data.iloc[train_idx].drop(columns=['is_fraud'])
        y_train = self.data.iloc[train_idx]['is_fraud']
        X_val = self.data.iloc[val_idx].drop(columns=['is_fraud'])
        y_val = self.data.iloc[val_idx]['is_fraud']
        
        logger.info(f"Fold {fold}: Train shape {X_train.shape}, Validation shape {X_val.shape}")
        return X_train, y_train, X_val, y_val
    
    def prepare_data(self, filepath: str, balance: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for model training.
        
        Args:
            filepath: Path to the processed CSV file
            balance: Whether to balance the classes
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Load data
        self.load_data(filepath)
        
        # Balance if requested
        if balance:
            self.data = self.balance_data()
            
        # Split into train and test sets
        train_df, test_df = self.train_test_split()
        
        # Create CV folds for training data
        self.create_cv_folds(train_df)
        
        return train_df, test_df