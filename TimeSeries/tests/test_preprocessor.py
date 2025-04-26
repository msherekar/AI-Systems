# Test for DataPreprocessor

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime
from src.data.preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file with test data
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'test_data.csv')
        
        # Create test data
        test_data = pd.DataFrame({
            'trans_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'trans_amount': [100.0, 200.0, 300.0],
            'is_fraud': [0, 1, 0]
        })
        
        test_data.to_csv(self.temp_file, index=False)
        
        # Create preprocessor
        self.preprocessor = DataPreprocessor(self.temp_file)
    
    def tearDown(self):
        # Clean up temporary files
        os.unlink(self.temp_file)
        os.rmdir(self.temp_dir)
    
    def test_load_data(self):
        # Test loading data
        df = self.preprocessor.load_data()
        
        # Check if DataFrame is returned
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check if all columns are present
        self.assertTrue('trans_date' in df.columns)
        self.assertTrue('trans_amount' in df.columns)
        self.assertTrue('is_fraud' in df.columns)
        
        # Check if data is loaded correctly
        self.assertEqual(len(df), 3)
    
    def test_preprocess(self):
        # Load data
        df = self.preprocessor.load_data()
        
        # Preprocess data
        processed_df = self.preprocessor.preprocess(df)
        
        # Check if trans_date is converted to datetime
        self.assertTrue(pd.api.types.is_datetime64_dtype(processed_df['trans_date']))
        
        # Check if trans_month is added
        self.assertIn('trans_month', processed_df.columns)
        
        # Check if trans_month is of period type
        self.assertTrue(isinstance(processed_df['trans_month'].iloc[0], pd.Period))
        
        # Check if the first date is processed correctly
        self.assertEqual(processed_df['trans_month'].iloc[0].strftime('%Y-%m'), '2023-01')

    def test_preprocess_with_missing_date(self):
        # Create test data with missing date
        test_data = pd.DataFrame({
            'trans_date': ['2023-01-01', None, '2023-01-03'],
            'trans_amount': [100.0, 200.0, 300.0],
            'is_fraud': [0, 1, 0]
        })
        
        try:
            # Preprocess data
            processed_df = self.preprocessor.preprocess(test_data)
            
            # Check if missing date was handled
            self.assertEqual(len(processed_df), 2)  # One row should be dropped
        except Exception as e:
            # If the function doesn't handle missing dates, this will catch the error
            self.fail(f"preprocess() raised {type(e).__name__} unexpectedly!")
    
    def test_preprocess_with_invalid_date_format(self):
        # Create test data with invalid date format
        test_data = pd.DataFrame({
            'trans_date': ['2023-01-01', 'invalid_date', '2023-01-03'],
            'trans_amount': [100.0, 200.0, 300.0],
            'is_fraud': [0, 1, 0]
        })
        
        try:
            # Preprocess data
            processed_df = self.preprocessor.preprocess(test_data)
            
            # Check if invalid date was handled
            self.assertEqual(len(processed_df), 2)  # One row should be dropped
        except Exception as e:
            # If the function doesn't handle invalid dates, this will catch the error
            self.fail(f"preprocess() raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main() 