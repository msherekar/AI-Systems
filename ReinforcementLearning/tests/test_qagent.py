import unittest
import numpy as np
import tempfile
import os
from src.models.qlearning_agent import QLearningAgent

class TestQLearningAgent(unittest.TestCase):
    """Unit tests for the QLearningAgent class."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.state_size = 100
        self.action_size = 3
        self.agent = QLearningAgent(self.state_size, self.action_size)
        self.test_state = ('Male', 'Premium', 35, 24)
        self.test_action = 1
        self.test_reward = 1.0
        self.test_next_state = ('Male', 'Premium', 35, 25)
        
    def test_init(self):
        """Test agent initialization with default parameters."""
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertEqual(self.agent.discount_factor, 0.9)
        self.assertEqual(self.agent.epsilon, 0.1)
        self.assertTrue(isinstance(self.agent.Q_table, np.ndarray))
        self.assertEqual(self.agent.Q_table.shape, (self.state_size, self.action_size))
        self.assertEqual(np.sum(self.agent.Q_table), 0.0)  # Initial Q-table should be all zeros
        
    def test_state_mapping(self):
        """Test consistent state mapping."""
        # First mapping should create an entry
        state_index = self.agent.map_state_to_index(self.test_state)
        self.assertTrue(0 <= state_index < self.state_size)
        
        # Second mapping of the same state should give the same index
        state_index2 = self.agent.map_state_to_index(self.test_state)
        self.assertEqual(state_index, state_index2)
        
        # Different state should map to different index
        different_state = ('Female', 'Basic', 28, 12)
        different_state_index = self.agent.map_state_to_index(different_state)
        self.assertNotEqual(state_index, different_state_index)
        
    def test_action_mapping(self):
        """Test action mapping."""
        # Action mapping should return the action modulo action_size
        self.assertEqual(self.agent.map_action_to_index(1), 1)
        self.assertEqual(self.agent.map_action_to_index(self.action_size + 1), 1)
        
    def test_choose_action(self):
        """Test action selection with epsilon-greedy policy."""
        # Force exploration by setting epsilon to 1
        self.agent.epsilon = 1.0
        action = self.agent.choose_action(self.test_state)
        self.assertTrue(0 <= action < self.action_size)
        
        # Force exploitation by setting epsilon to 0
        self.agent.epsilon = 0.0
        
        # Update Q-table to have a clear best action
        state_index = self.agent.map_state_to_index(self.test_state)
        self.agent.Q_table[state_index, 0] = 1.0
        self.agent.Q_table[state_index, 1] = 2.0
        self.agent.Q_table[state_index, 2] = 0.5
        
        action = self.agent.choose_action(self.test_state)
        self.assertEqual(action, 1)  # Should choose action with highest Q-value
        
    def test_update_q_table(self):
        """Test Q-table update functionality."""
        # Get initial state and action indices
        state_index = self.agent.map_state_to_index(self.test_state)
        action_index = self.agent.map_action_to_index(self.test_action)
        
        # Initial Q-value should be 0
        initial_q_value = self.agent.Q_table[state_index, action_index]
        self.assertEqual(initial_q_value, 0.0)
        
        # Update Q-table
        self.agent.update_q_table(self.test_state, self.test_action, self.test_reward, self.test_next_state)
        
        # Q-value should have changed
        updated_q_value = self.agent.Q_table[state_index, action_index]
        self.assertNotEqual(updated_q_value, initial_q_value)
        
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Update Q-table with some values
        self.agent.update_q_table(self.test_state, self.test_action, self.test_reward, self.test_next_state)
        
        # Create temporary file for saving
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_filename = temp_file.name
            
        try:
            # Save agent
            self.agent.save(temp_filename)
            
            # Load agent
            loaded_agent = QLearningAgent.load(temp_filename)
            
            # Check that loaded agent has the same parameters
            self.assertEqual(loaded_agent.state_size, self.agent.state_size)
            self.assertEqual(loaded_agent.action_size, self.agent.action_size)
            self.assertEqual(loaded_agent.learning_rate, self.agent.learning_rate)
            self.assertEqual(loaded_agent.discount_factor, self.agent.discount_factor)
            self.assertEqual(loaded_agent.epsilon, self.agent.epsilon)
            
            # Check that Q-table values are preserved
            state_index = self.agent.map_state_to_index(self.test_state)
            action_index = self.agent.map_action_to_index(self.test_action)
            self.assertEqual(loaded_agent.Q_table[state_index, action_index], 
                             self.agent.Q_table[state_index, action_index])
            
            # Check that state mapping is preserved
            self.assertEqual(loaded_agent.map_state_to_index(self.test_state),
                            self.agent.map_state_to_index(self.test_state))
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
if __name__ == '__main__':
    unittest.main() 