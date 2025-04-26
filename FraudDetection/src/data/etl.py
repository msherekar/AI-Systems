# src/data/etl.py
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class ETLPipeline:
    """Data extraction, transformation and loading pipeline for fraud detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ETL pipeline.
        
        Args:
            config: Dictionary containing ETL configuration
        """
        self.config = config
        self.night_start_hour = config.get('preprocessing', {}).get('night_start_hour', 22)
        self.night_end_hour = config.get('preprocessing', {}).get('night_end_hour', 4)
        self.raw_dir = config.get('data', {}).get('raw_dir', 'data/raw')
        self.processed_dir = config.get('data', {}).get('processed_dir', 'data/processed')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def extract(self, filepath: str) -> pd.DataFrame:
        """
        Extract data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame containing the extracted data
        """
        if not filepath.endswith('.csv'):
            raise ValueError(f"File {filepath} is not a CSV file")
        
        try:
            logger.info(f"Extracting data from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Extracted {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error extracting data from {filepath}: {e}")
            raise
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-related features to the DataFrame.
        
        Args:
            df: DataFrame containing transaction data with 'unix_time' column
            
        Returns:
            DataFrame with time-related features added
        """
        logger.info("Adding time-related features")
        
        # Create a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Convert Unix timestamp to datetime
        result['datetime'] = pd.to_datetime(result['unix_time'], unit='s')
        
        # Extract hour
        result['hour'] = result['datetime'].dt.hour
        
        # Create is_night feature
        result['is_night'] = ((result['hour'] >= self.night_start_hour) | 
                               (result['hour'] < self.night_end_hour)).astype(int)
        
        return result
    
    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar-related features to the DataFrame.
        
        Args:
            df: DataFrame containing transaction data with 'datetime' column
            
        Returns:
            DataFrame with calendar-related features added
        """
        logger.info("Adding calendar-related features")
        
        # Create a copy to avoid SettingWithCopyWarning
        result = df.copy()
        
        # Define boolean masks for each period
        holidays_mask = ((result['datetime'].dt.month == 12) & (result['datetime'].dt.day >= 24)) | \
                        ((result['datetime'].dt.month == 1) & (result['datetime'].dt.day <= 1))
                        
        post_holidays_mask = (result['datetime'].dt.month == 1) | (result['datetime'].dt.month == 2)
        summer_mask = (result['datetime'].dt.month >= 5) & (result['datetime'].dt.month <= 9)

        # Create one-hot encoded features
        result['is_holidays'] = holidays_mask.astype(int)
        result['is_post_holidays'] = post_holidays_mask.astype(int)
        result['is_summer'] = summer_mask.astype(int)
        
        return result
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant features for the model.
        
        Args:
            df: Input DataFrame with all features
            
        Returns:
            DataFrame with selected features
        """
        logger.info("Selecting relevant features")
        
        # Define important columns
        imp_columns = ['amt', 'cc_num', 'is_night', 'is_holidays', 
                       'is_post_holidays', 'is_summer']
        
        # Add target column if it exists
        if 'is_fraud' in df.columns:
            imp_columns.append('is_fraud')
            
        # Select columns
        return df[imp_columns]
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformation steps to the input DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info(f"Transforming data with shape {df.shape}")
        
        # Apply transformation pipeline
        df = self.add_time_features(df)
        df = self.add_calendar_features(df)
        df = self.select_features(df)
        
        logger.info(f"Transformed data to shape {df.shape}")
        return df
    
    def save(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save transformed DataFrame to CSV.
        
        Args:
            df: DataFrame to save
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        output_path = os.path.join(self.processed_dir, filename)
        
        logger.info(f"Saving transformed data to {output_path}")
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def process_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Process a single file through the ETL pipeline.
        
        Args:
            input_file: Path to the input CSV file
            output_file: Name of the output file. If None, derive from input file.
            
        Returns:
            Path to the processed file
        """
        # If output_file is not provided, derive from input_file
        if output_file is None:
            output_file = os.path.basename(input_file).replace('.csv', '_processed.csv')
        
        logger.info(f"Processing file {input_file} to {output_file}")
        
        # Extract
        df = self.extract(input_file)
        
        # Transform
        transformed_df = self.transform(df)
        
        # Load
        output_path = self.save(transformed_df, output_file)
        
        return output_path