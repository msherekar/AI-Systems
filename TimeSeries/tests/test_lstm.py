# Test for LSTM models

import unittest
import torch
import numpy as np
from src.models.lstm_total import LSTMTotalModel
from src.models.lstm_fraud import LSTMFraudModel

class TestLSTMModels(unittest.TestCase):
    def setUp(self):
        self.input_size = 1
        self.hidden_size = 50
        self.num_layers = 1
        self.batch_size = 10
        self.seq_length = 7
        
        # Create models
        self.total_model = LSTMTotalModel(self.input_size, self.hidden_size, self.num_layers)
        self.fraud_model = LSTMFraudModel(self.input_size, self.hidden_size, self.num_layers)
        
        # Set models to evaluation mode
        self.total_model.eval()
        self.fraud_model.eval()
        
        # Create random input data
        self.random_input = torch.randn(self.batch_size, self.seq_length, self.input_size)

    def test_total_model_forward(self):
        # Test the forward pass of the total model
        with torch.no_grad():
            output = self.total_model(self.random_input)
        
        # Check the output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Check that output is a tensor
        self.assertIsInstance(output, torch.Tensor)
    
    def test_fraud_model_forward(self):
        # Test the forward pass of the fraud model
        with torch.no_grad():
            output = self.fraud_model(self.random_input)
        
        # Check the output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Check that output is a tensor
        self.assertIsInstance(output, torch.Tensor)
    
    def test_model_training(self):
        # Create dummy training data
        X = torch.randn(self.batch_size, self.seq_length, self.input_size)
        y = torch.randn(self.batch_size, 1)
        
        # Create loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.total_model.parameters(), lr=0.001)
        
        # Set model to training mode
        self.total_model.train()
        
        # Initial forward pass
        output = self.total_model(X)
        initial_loss = criterion(output, y)
        
        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            output = self.total_model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # Final forward pass
        output = self.total_model(X)
        final_loss = criterion(output, y)
        
        # Check that the loss decreased
        self.assertLess(final_loss.item(), initial_loss.item())
    
    def test_model_save_load(self):
        # Get model state dict
        state_dict = self.total_model.state_dict()
        
        # Create a new model
        new_model = LSTMTotalModel(self.input_size, self.hidden_size, self.num_layers)
        
        # Load state dict into new model
        new_model.load_state_dict(state_dict)
        
        # Set both models to eval mode
        self.total_model.eval()
        new_model.eval()
        
        # Forward pass on both models
        with torch.no_grad():
            original_output = self.total_model(self.random_input)
            new_output = new_model(self.random_input)
        
        # Check that the outputs are the same
        self.assertTrue(torch.allclose(original_output, new_output))
    
    def test_batch_processing(self):
        # Test that models can handle batched data properly
        
        # Create batches of different sizes
        small_batch = torch.randn(2, self.seq_length, self.input_size)
        large_batch = torch.randn(20, self.seq_length, self.input_size)
        
        # Forward pass with different batch sizes
        with torch.no_grad():
            output_small = self.total_model(small_batch)
            output_large = self.total_model(large_batch)
        
        # Check output shapes
        self.assertEqual(output_small.shape, (2, 1))
        self.assertEqual(output_large.shape, (20, 1))
    
    def test_varying_sequence_length(self):
        # Test that models can handle varying sequence lengths
        
        # Create sequences of different lengths
        short_seq = torch.randn(self.batch_size, 5, self.input_size)
        long_seq = torch.randn(self.batch_size, 10, self.input_size)
        
        # Forward pass with different sequence lengths
        with torch.no_grad():
            output_short = self.total_model(short_seq)
            output_long = self.total_model(long_seq)
        
        # Check output shapes
        self.assertEqual(output_short.shape, (self.batch_size, 1))
        self.assertEqual(output_long.shape, (self.batch_size, 1))

if __name__ == '__main__':
    unittest.main() 