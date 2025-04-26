# Test for ConfigManager

import unittest
import os
import tempfile
import yaml
from pathlib import Path
from src.utils.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a temporary config file
        self.config_dir = os.path.join(self.temp_dir.name, 'config')
        os.makedirs(self.config_dir, exist_ok=True)
        
        self.config_path = os.path.join(self.config_dir, 'config.yaml')
        
        # Define a sample configuration
        self.sample_config = {
            'data': {
                'raw_data_path': 'data/raw/credit_card_data.csv',
                'processed_data_path': 'data/processed/processed_data.csv'
            },
            'models': {
                'sarimax': {
                    'total': {
                        'order': [1, 1, 1],
                        'seasonal_order': [1, 1, 1, 12]
                    },
                    'fraud': {
                        'order': [1, 1, 1],
                        'seasonal_order': [1, 1, 1, 12]
                    }
                },
                'lstm': {
                    'total': {
                        'input_size': 1,
                        'hidden_size': 50,
                        'num_layers': 1,
                        'epochs': 30,
                        'batch_size': 32
                    },
                    'fraud': {
                        'input_size': 1,
                        'hidden_size': 50,
                        'num_layers': 1,
                        'epochs': 30,
                        'batch_size': 32
                    }
                }
            },
            'training': {
                'train_test_split': 0.8,
                'random_state': 42
            },
            'api': {
                'host': '0.0.0.0',
                'port': 80
            }
        }
        
        # Write the configuration to the file
        with open(self.config_path, 'w') as f:
            yaml.dump(self.sample_config, f)
        
        # Set up the config manager
        # Note: This would normally use the mocked file, but for testing
        # purposes we'll directly set the config
        self.config_manager = ConfigManager()
        self.config_manager.config = self.sample_config
    
    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()
    
    def test_get_data_paths(self):
        # Test getting data paths
        data_paths = self.config_manager.get_data_paths()
        
        # Check that the data paths match the sample configuration
        self.assertEqual(data_paths, self.sample_config['data'])
        
        # Check specific path values
        self.assertEqual(data_paths['raw_data_path'], 'data/raw/credit_card_data.csv')
        self.assertEqual(data_paths['processed_data_path'], 'data/processed/processed_data.csv')
    
    def test_get_model_params(self):
        # Test getting model parameters for SARIMAX total model
        sarimax_total_params = self.config_manager.get_model_params('sarimax', 'total')
        
        # Check that the parameters match the sample configuration
        self.assertEqual(sarimax_total_params, self.sample_config['models']['sarimax']['total'])
        
        # Check specific parameter values
        self.assertEqual(sarimax_total_params['order'], [1, 1, 1])
        self.assertEqual(sarimax_total_params['seasonal_order'], [1, 1, 1, 12])
        
        # Test getting model parameters for LSTM fraud model
        lstm_fraud_params = self.config_manager.get_model_params('lstm', 'fraud')
        
        # Check that the parameters match the sample configuration
        self.assertEqual(lstm_fraud_params, self.sample_config['models']['lstm']['fraud'])
        
        # Check specific parameter values
        self.assertEqual(lstm_fraud_params['input_size'], 1)
        self.assertEqual(lstm_fraud_params['hidden_size'], 50)
        self.assertEqual(lstm_fraud_params['batch_size'], 32)
    
    def test_get_training_params(self):
        # Test getting training parameters
        training_params = self.config_manager.get_training_params()
        
        # Check that the parameters match the sample configuration
        self.assertEqual(training_params, self.sample_config['training'])
        
        # Check specific parameter values
        self.assertEqual(training_params['train_test_split'], 0.8)
        self.assertEqual(training_params['random_state'], 42)
    
    def test_get_api_settings(self):
        # Test getting API settings
        api_settings = self.config_manager.get_api_settings()
        
        # Check that the settings match the sample configuration
        self.assertEqual(api_settings, self.sample_config['api'])
        
        # Check specific setting values
        self.assertEqual(api_settings['host'], '0.0.0.0')
        self.assertEqual(api_settings['port'], 80)
    
    def test_nonexistent_model_params(self):
        # Test getting parameters for a model that doesn't exist
        nonexistent_params = self.config_manager.get_model_params('nonexistent', 'total')
        
        # Should return an empty dictionary
        self.assertEqual(nonexistent_params, {})
    
    def test_nonexistent_config_section(self):
        # Create a config manager with a minimal configuration
        minimal_config = {'data': {'raw_data_path': 'data/raw.csv'}}
        config_manager = ConfigManager()
        config_manager.config = minimal_config
        
        # Test getting a section that doesn't exist
        api_settings = config_manager.get_api_settings()
        
        # Should return an empty dictionary
        self.assertEqual(api_settings, {})

if __name__ == '__main__':
    unittest.main() 