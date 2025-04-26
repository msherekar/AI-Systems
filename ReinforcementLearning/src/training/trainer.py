"""
Training orchestration for the reinforcement learning model.

This module contains the Trainer class that orchestrates the training
of the Q-learning agent.
"""

import numpy as np
import pandas as pd
import logging
import time
import os
from typing import List, Tuple, Dict, Any, Optional, Union
from src.data.preprocessor import DataPreprocessor
from src.data.state_action import StateActionHandler
from src.models.qlearning_agent import QLearningAgent
from src.models.metrics import Metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for the Q-learning agent.
    
    This class orchestrates the training process, including data preprocessing,
    state and action generation, agent training, and metric calculation.
    """
    
    def __init__(self, 
                 agent_params: Dict[str, Any] = None,
                 output_dir: str = 'models',
                 metrics_dir: str = '.',
                 logs_dir: str = 'logs'):
        """
        Initialize the trainer.
        
        Args:
            agent_params: Parameters for the Q-learning agent
            output_dir: Directory to save the trained model
            metrics_dir: Directory to save metric plots
            logs_dir: Directory to save training logs
        """
        self.agent_params = agent_params or {}
        self.output_dir = output_dir
        self.metrics_dir = metrics_dir
        self.logs_dir = logs_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.state_action_handler = StateActionHandler()
        self.metrics = Metrics(output_dir=metrics_dir)
        self.agent = None
    
    def preprocess_data(self, userbase_file: str, sent_file: str, responded_file: str) -> pd.DataFrame:
        """
        Preprocess the data for training.
        
        Args:
            userbase_file: Path to the userbase CSV file
            sent_file: Path to the sent emails CSV file
            responded_file: Path to the responded emails CSV file
            
        Returns:
            Processed data ready for training
        """
        logger.info("Preprocessing data...")
        processed_data = self.preprocessor.preprocess_for_training(userbase_file, sent_file, responded_file)
        logger.info(f"Processed data shape: {processed_data.shape}")
        return processed_data
    
    def generate_states_and_actions(self, data: pd.DataFrame) -> Tuple[List[Tuple], np.ndarray]:
        """
        Generate states and actions from the processed data.
        
        Args:
            data: Processed data
            
        Returns:
            Tuple of (states, actions)
        """
        logger.info("Generating states and actions...")
        states = self.state_action_handler.generate_states(data)
        actions = self.state_action_handler.generate_actions(data)
        logger.info(f"Generated {len(states)} states and {len(actions)} actions")
        return states, actions
    
    def initialize_agent(self, state_size: int, action_size: int) -> QLearningAgent:
        """
        Initialize the Q-learning agent.
        
        Args:
            state_size: Number of possible states or size of state space
            action_size: Number of possible actions
            
        Returns:
            Initialized Q-learning agent
        """
        logger.info("Initializing Q-learning agent...")
        learning_rate = self.agent_params.get('learning_rate', 0.1)
        discount_factor = self.agent_params.get('discount_factor', 0.9)
        epsilon = self.agent_params.get('epsilon', 0.1)
        
        agent = QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon
        )
        
        logger.info(f"Agent initialized with parameters: learning_rate={learning_rate}, "
                    f"discount_factor={discount_factor}, epsilon={epsilon}")
        return agent
    
    def train(self, data: pd.DataFrame, episodes: int = 1000) -> Dict[str, Any]:
        """
        Train the Q-learning agent.
        
        Args:
            data: Processed data
            episodes: Number of training episodes
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Starting training for {episodes} episodes...")
        
        # Generate states and actions
        states, actions = self.generate_states_and_actions(data)
        
        # Initialize agent
        self.agent = self.initialize_agent(len(states), len(actions))
        
        # Training loop
        rewards_per_episode = []
        training_start_time = time.time()
        
        for episode in range(episodes):
            episode_rewards = []
            
            # Iterate through all states
            for state_idx, state in enumerate(states):
                # Choose action using epsilon-greedy policy
                action_idx = self.agent.choose_action(state)
                action = actions[action_idx]
                
                # Get reward for this state-action pair
                reward = self.state_action_handler.get_reward(state, data)
                episode_rewards.append(reward)
                
                # Get next state (simple transition to next state in the list, or wrap around)
                next_state_idx = (state_idx + 1) % len(states)
                next_state = states[next_state_idx]
                
                # Update Q-table
                self.agent.train(state, action, reward, next_state)
            
            # Store average reward for this episode
            rewards_per_episode.append(np.mean(episode_rewards))
            
            # Log progress periodically
            if (episode + 1) % 100 == 0 or episode == 0:
                logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {rewards_per_episode[-1]:.4f}")
        
        training_time = time.time() - training_start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Calculate metrics
        metrics = self.metrics.plot_all_metrics(rewards_per_episode, self.agent.discount_factor)
        
        # Save agent
        model_path = os.path.join(self.output_dir, 'q_table.pkl')
        self.agent.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save training log
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        log_path = os.path.join(self.logs_dir, f"training_{timestamp}.txt")
        with open(log_path, 'w') as f:
            f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Episodes: {episodes}\n")
            f.write(f"Final average reward: {rewards_per_episode[-1]:.4f}\n")
            f.write(f"Episodes to convergence: {metrics['episodes_to_convergence']}\n")
            f.write(f"Training time: {training_time:.2f} seconds\n")
        
        return {
            'agent': self.agent,
            'rewards_per_episode': rewards_per_episode,
            'metrics': metrics,
            'training_time': training_time
        }
    
    def load_agent(self, model_path: str) -> QLearningAgent:
        """
        Load a trained agent from a file.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded Q-learning agent
        """
        logger.info(f"Loading agent from {model_path}...")
        self.agent = QLearningAgent.load(model_path)
        return self.agent 