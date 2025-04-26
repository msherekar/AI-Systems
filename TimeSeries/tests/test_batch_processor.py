# Test for BatchProcessor

import unittest
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging
from src.utils.batch_processor import BatchProcessor

class TestBatchProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test CSV file with sample data
        self.test_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100),
            'value': np.random.rand(100) * 1000,
            'category': np.random.choice(['A', 'B', 'C'], size=100)
        })
        
        self.test_file = os.path.join(self.temp_dir, 'test_data.csv')
        self.test_data.to_csv(self.test_file, index=False)
        
        # Create a logger for testing
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)
        
        # Initialize the BatchProcessor
        self.processor = BatchProcessor(file_path=self.test_file, batch_size=10, logger=self.logger)
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test BatchProcessor initialization."""
        self.assertEqual(self.processor.file_path, self.test_file)
        self.assertEqual(self.processor.batch_size, 10)
        self.assertEqual(self.processor.logger, self.logger)
    
    def test_process_in_batches(self):
        """Test processing data in batches."""
        # Define a simple processing function
        def process_func(df):
            return df['value'].sum()
        
        # Process the data and get results
        results = self.processor.process_in_batches(process_func=process_func)
        
        # Check if the results match the expected sum
        expected_sum = self.test_data['value'].sum()
        self.assertAlmostEqual(sum(results), expected_sum, places=5)
    
    def test_process_in_batches_with_output_file(self):
        """Test processing data in batches with output file."""
        output_file = os.path.join(self.temp_dir, 'output.csv')
        
        # Define a processing function that doubles the values
        def process_func(df):
            df['value'] = df['value'] * 2
            return df
        
        # Process the data and save to output file
        self.processor.process_in_batches(
            process_func=process_func,
            output_path=output_file
        )
        
        # Check if the output file exists and has the expected values
        self.assertTrue(os.path.exists(output_file))
        
        # Read the output file
        output_data = pd.read_csv(output_file)
        
        # Check if all rows are processed
        self.assertEqual(len(output_data), len(self.test_data))
        
        # Check if values are doubled
        original_values = self.test_data['value'].values
        processed_values = output_data['value'].values
        np.testing.assert_array_almost_equal(processed_values, original_values * 2)
    
    def test_process_time_series_in_batches(self):
        """Test processing time series data in batches with aggregation."""
        # Process the time series data with sum aggregation
        result = self.processor.process_time_series_in_batches(
            date_column='date',
            values_column='value',
            aggregation='sum',
            group_by='category'
        )
        
        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if the result contains all categories
        self.assertEqual(set(result['category'].unique()), set(self.test_data['category'].unique()))
        
        # Manually compute the expected sums for each category and compare
        for category in self.test_data['category'].unique():
            expected_sum = self.test_data[self.test_data['category'] == category]['value'].sum()
            category_sum = result[result['category'] == category]['value'].sum()
            self.assertAlmostEqual(category_sum, expected_sum, places=5)
    
    def test_invalid_file_path(self):
        """Test handling of invalid file path."""
        processor = BatchProcessor(file_path='nonexistent.csv', batch_size=10)
        
        # Define a simple processing function
        def process_func(df):
            return df['value'].sum()
        
        # The process_in_batches method should raise a FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            processor.process_in_batches(process_func=process_func)
    
    def test_empty_batch(self):
        """Test handling of empty batches."""
        # Create an empty CSV file
        empty_file = os.path.join(self.temp_dir, 'empty.csv')
        pd.DataFrame(columns=['date', 'value']).to_csv(empty_file, index=False)

        processor = BatchProcessor(file_path=empty_file, batch_size=10)

        # Define a simple processing function
        def process_func(df):
            return pd.DataFrame() if df.empty else df['value'].sum()

        # Process the empty file
        results = processor.process_in_batches(process_func=process_func)

        # Check if the result is an empty DataFrame
        self.assertTrue(results.empty)



if __name__ == '__main__':
    unittest.main() 