"""
Data preprocessing utilities for the Reinforcement Learning model.

This module contains the DataPreprocessor class which handles loading,
merging, and preprocessing data for the Q-learning agent.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Optional, Dict, Any
from io import BytesIO


class DataPreprocessor:
    """
    Class to preprocess data and obtain states to be fed into the Q-table.
    
    This class handles all data preprocessing tasks including loading, merging,
    reward calculation, and state extraction.
    """
    
    def __init__(self, important_features=None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            important_features: List of features to use for state generation.
                                If None, will use default features.
        """
        self.important_features = important_features or ['Gender', 'Type', 'Age', 'Tenure']
    
    def load_data(self, userbase_file: str, sent_file: str, responded_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the userbase, sent emails, and responded emails data from CSV files.
        
        Args:
            userbase_file: Path to the userbase CSV file
            sent_file: Path to the sent emails CSV file
            responded_file: Path to the responded emails CSV file
            
        Returns:
            Tuple of DataFrames: (userbase, sent_emails, responded_emails)
        """
        userbase = pd.read_csv(userbase_file)
        sent = pd.read_csv(sent_file)
        responded = pd.read_csv(responded_file).drop_duplicates()
        return userbase, sent, responded
    
    def merge_data(self, sent: pd.DataFrame, userbase: pd.DataFrame, responded: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the sent emails data with the userbase and responded emails data,
        keeping only the important features along with Customer ID.
        
        Args:
            sent: DataFrame containing sent email information
            userbase: DataFrame containing user information
            responded: DataFrame containing response information
            
        Returns:
            DataFrame: Merged data with all relevant information
        """
        # Select important features along with Customer ID
        userbase_selected = userbase[['Customer_ID'] + self.important_features]

        # Merge the selected userbase data with sent emails
        merged_data = pd.merge(sent, userbase_selected, on='Customer_ID', how='left')

        # Fill missing values with corresponding values from userbase
        for feature in self.important_features:
            merged_data[feature] = merged_data[feature].fillna(userbase_selected.set_index('Customer_ID')[feature])

        # Merge with responded emails
        merged_data = pd.merge(merged_data, responded, on=['Customer_ID'], how='left')

        return merged_data
    
    def calculate_rewards(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rewards based on whether a customer responded to an email or not.
        
        Args:
            data: DataFrame with merged data
            
        Returns:
            DataFrame: Data with reward column added
        """
        data['Reward'] = np.where(data['Sent_Date'] == data['Responded_Date'], 1, 0)
        data['Responded_Date'].fillna(pd.to_datetime('1900-01-01'), inplace=True)
        data.rename(columns={'SubjectLine_ID_x': 'SubLine_Sent', 'SubjectLine_ID_y': 'SubLine_Responded'}, inplace=True)
        data['SubLine_Responded'].fillna(-1, inplace=True)
        return data
    
    def preprocess_for_training(self, userbase_file: str, sent_file: str, responded_file: str) -> pd.DataFrame:
        """
        Complete preprocessing workflow for training data.
        
        Args:
            userbase_file: Path to the userbase CSV file
            sent_file: Path to the sent emails CSV file
            responded_file: Path to the responded emails CSV file
            
        Returns:
            DataFrame: Processed data ready for training
        """
        userbase, sent, responded = self.load_data(userbase_file, sent_file, responded_file)
        merged_data = self.merge_data(sent, userbase, responded)
        processed_data = self.calculate_rewards(merged_data)
        return processed_data
    
    def extract_states_from_df(self, data: pd.DataFrame) -> List[Tuple]:
        """
        Extract states from preprocessed DataFrame.
        
        Args:
            data: DataFrame with preprocessed data
            
        Returns:
            List[Tuple]: List of state tuples
        """
        # Drop rows with missing values in important features
        data = data.dropna(subset=self.important_features)
        
        # Define states
        states = []
        
        # Iterate over rows
        for _, row in data.iterrows():
            # Extract features
            state_values = tuple(row[feature] for feature in self.important_features)
            
            # Append state to list of states
            states.append(state_values)
            
        return states
    
    def save_processed_data(self, data: pd.DataFrame, output_path: str) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            data: Processed DataFrame
            output_path: Path to save the CSV file
        """
        data.to_csv(output_path, index=False)
        
    def load_processed_data(self, input_path: str) -> pd.DataFrame:
        """
        Load processed data from CSV file.
        
        Args:
            input_path: Path to the CSV file
            
        Returns:
            DataFrame: Loaded data
        """
        return pd.read_csv(input_path)


def preprocess_data(file_obj: Any) -> List[Tuple]:
    """
    Preprocess a CSV file containing customer data.
    
    Args:
        file_obj: A file-like object containing CSV data
        
    Returns:
        List of tuples representing the preprocessed states
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_obj)
        
        # Check required columns
        required_columns = ['Gender', 'Type', 'Age', 'Tenure']
        if not all(col in df.columns for col in required_columns):
            return []
        
        # Convert rows to state tuples
        states = []
        for _, row in df.iterrows():
            state = (
                row['Gender'],
                row['Type'],
                row['Age'],
                row['Tenure']
            )
            states.append(state)
            
        return states
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        return [] 