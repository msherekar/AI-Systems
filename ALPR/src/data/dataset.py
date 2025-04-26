import numpy as np
import logging
from pathlib import Path
import yaml
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Dataset')

class ObjectDetectionDataset:
    """
    Class for managing object detection datasets, including data loading, splitting,
    and cross-validation.
    """
    
    def __init__(self, data=None, config_path='../../config/config.yaml'):
        """
        Initialize the ObjectDetectionDataset class.
        
        Args:
            data (numpy.ndarray): Optional dataset to initialize with.
            config_path (str): Path to configuration file.
        """
        self.data = data
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """
        Load configuration from YAML file.
        
        Returns:
            dict: Configuration parameters.
        """
        try:
            config_path = Path(self.config_path)
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def load_coco_annotations(self, json_path):
        """
        Load annotations from a COCO format JSON file.
        
        Args:
            json_path (str): Path to COCO annotations JSON file.
            
        Returns:
            list: List of annotations.
        """
        try:
            with open(json_path, 'r') as f:
                annotations = json.load(f)
                
            logger.info(f"Loaded {len(annotations['annotations'])} annotations from {json_path}")
            self.data = annotations
            return annotations
        except Exception as e:
            logger.error(f"Error loading COCO annotations: {e}")
            return None
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split the dataset into training, validation, and testing sets.

        Args:
            train_ratio (float): The ratio of training data. Default is 0.7.
            val_ratio (float): The ratio of validation data. Default is 0.15.
            test_ratio (float): The ratio of testing data. Default is 0.15.

        Returns:
            tuple: A tuple containing training, validation, and testing datasets.
        """
        if self.data is None:
            logger.error("No data available to split")
            return None, None, None
            
        data_size = len(self.data)
        indices = np.arange(data_size)
        np.random.shuffle(indices)

        train_end = int(data_size * train_ratio)
        val_end = int(data_size * (train_ratio + val_ratio))

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        if isinstance(self.data, np.ndarray):
            train_data = self.data[train_indices]
            val_data = self.data[val_indices]
            test_data = self.data[test_indices]
        else:
            train_data = [self.data[i] for i in train_indices]
            val_data = [self.data[i] for i in val_indices]
            test_data = [self.data[i] for i in test_indices]

        logger.info(f"Split dataset: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        return train_data, val_data, test_data

    def k_fold_split(self, k=5):
        """
        Split the dataset into k folds for cross-validation.

        Args:
            k (int): The number of folds for cross-validation.

        Returns:
            list: A list of tuples containing train and validation indices for each fold.
        """
        if self.data is None:
            logger.error("No data available for k-fold split")
            return []
            
        data_size = len(self.data)
        fold_size = data_size // k
        remainder = data_size % k

        indices = np.arange(data_size)
        np.random.shuffle(indices)

        folds = []
        start = 0
        for i in range(k):
            end = start + fold_size
            if i < remainder:
                end += 1
                
            # Validation indices for this fold
            val_indices = indices[start:end]
            
            # Training indices for this fold (all indices except validation)
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            folds.append((train_indices, val_indices))
            start = end

        logger.info(f"Created {k} folds for cross-validation")
        return folds
    
    def get_fold_data(self, fold_index, folds):
        """
        Get the training and validation data for a specific fold.
        
        Args:
            fold_index (int): Index of the fold to retrieve.
            folds (list): List of fold indices from k_fold_split.
            
        Returns:
            tuple: Training and validation data for the specified fold.
        """
        if fold_index < 0 or fold_index >= len(folds):
            logger.error(f"Invalid fold index {fold_index}, must be between 0 and {len(folds)-1}")
            return None, None
            
        train_indices, val_indices = folds[fold_index]
        
        if isinstance(self.data, np.ndarray):
            train_data = self.data[train_indices]
            val_data = self.data[val_indices]
        else:
            train_data = [self.data[i] for i in train_indices]
            val_data = [self.data[i] for i in val_indices]
            
        logger.info(f"Fold {fold_index}: train={len(train_data)}, val={len(val_data)}")
        return train_data, val_data
    
    def save_dataset_split(self, train_data, val_data, test_data, output_dir):
        """
        Save the dataset splits to disk.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            test_data: Testing data.
            output_dir (str): Directory to save the splits.
            
        Returns:
            bool: True if saved successfully, False otherwise.
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            
            if isinstance(train_data, np.ndarray):
                np.save(output_path / "train_data.npy", train_data)
                np.save(output_path / "val_data.npy", val_data)
                np.save(output_path / "test_data.npy", test_data)
            else:
                with open(output_path / "train_data.json", 'w') as f:
                    json.dump(train_data, f)
                with open(output_path / "val_data.json", 'w') as f:
                    json.dump(val_data, f)
                with open(output_path / "test_data.json", 'w') as f:
                    json.dump(test_data, f)
                    
            logger.info(f"Dataset splits saved to {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Error saving dataset splits: {e}")
            return False
    
    def load_dataset_split(self, input_dir, format='json'):
        """
        Load dataset splits from disk.
        
        Args:
            input_dir (str): Directory containing the splits.
            format (str): File format ('json' or 'npy').
            
        Returns:
            tuple: Training, validation, and testing data.
        """
        try:
            input_path = Path(input_dir)
            
            if format == 'npy':
                train_data = np.load(input_path / "train_data.npy")
                val_data = np.load(input_path / "val_data.npy")
                test_data = np.load(input_path / "test_data.npy")
            else:
                with open(input_path / "train_data.json", 'r') as f:
                    train_data = json.load(f)
                with open(input_path / "val_data.json", 'r') as f:
                    val_data = json.load(f)
                with open(input_path / "test_data.json", 'r') as f:
                    test_data = json.load(f)
                    
            logger.info(f"Loaded dataset splits from {input_dir}")
            return train_data, val_data, test_data
        except Exception as e:
            logger.error(f"Error loading dataset splits: {e}")
            return None, None, None
