import unittest
import json
import io
import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, jsonify
import sys
import tempfile

# Import service for testing
from src.api.service import app, EmailMarketingService

class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints in the Flask application."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a mock Q-table for testing
        self.mock_q_table = np.zeros((100, 3))
        
        # Create a temporary file for the mock Q-table
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        with open(self.temp_file.name, 'wb') as f:
            pickle.dump(self.mock_q_table, f)
        
        # Create a mock service with the temporary file
        app.config['TESTING'] = True
        app.config['MODEL_PATH'] = self.temp_file.name
        
        # Force reload of the service with the new model path
        app.testing_service = EmailMarketingService(model_path=self.temp_file.name)
        
        # Create Flask test client
        self.client = app.test_client()
        
        # Create sample test data
        data = {
            'Gender': ['Male', 'Female'],
            'Type': ['Premium', 'Basic'],
            'Age': [35, 28],
            'Tenure': [24, 12]
        }
        self.test_df = pd.DataFrame(data)
        
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        
    def test_suggest_subject_lines_no_file(self):
        """Test the suggest_subject_lines endpoint with no file."""
        response = self.client.post('/suggest_subject_lines')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
    def test_suggest_subject_lines_empty_filename(self):
        """Test the suggest_subject_lines endpoint with empty filename."""
        response = self.client.post(
            '/suggest_subject_lines',
            data={'new_state': (io.BytesIO(), '')}
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
    def test_suggest_subject_lines_wrong_extension(self):
        """Test the suggest_subject_lines endpoint with wrong file extension."""
        response = self.client.post(
            '/suggest_subject_lines',
            data={'new_state': (io.BytesIO(b'data'), 'test.txt')}
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
    def test_suggest_subject_lines_success(self):
        """Test the suggest_subject_lines endpoint with valid data."""
        # Create CSV data from test DataFrame
        csv_data = self.test_df.to_csv(index=False).encode('utf-8')
        
        # Send request with CSV file
        response = self.client.post(
            '/suggest_subject_lines',
            data={'new_state': (io.BytesIO(csv_data), 'test.csv')},
            content_type='multipart/form-data'
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(isinstance(data, dict))
        
        # Check that we have responses for each state
        self.assertEqual(len(data), len(self.test_df))
        
        # Check format of each response
        for state, subject_line in data.items():
            self.assertTrue(isinstance(state, str))
            self.assertTrue(isinstance(subject_line, str))
            
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary file
        os.remove(self.temp_file.name)
        
        # Clean up app testing attributes
        if hasattr(app, 'testing_service'):
            delattr(app, 'testing_service')
            
if __name__ == '__main__':
    unittest.main() 