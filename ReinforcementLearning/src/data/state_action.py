"""
State and Action generation utilities for the Reinforcement Learning model.

This module contains functions for generating states and actions from
preprocessed data and calculating rewards.
"""

from itertools import product
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np


class StateActionHandler:
    """
    Handler for generating states, actions, and rewards from data.
    
    This class provides methods to generate states and actions, and to 
    calculate rewards for the Q-learning model.
    """
    
    def __init__(self):
        """Initialize the StateActionHandler."""
        self.states = None
        self.actions = None
        self.state_map = {}
    
    def generate_states(self, data: pd.DataFrame, features: List[str] = None) -> List[Tuple]:
        """
        Generate states based on unique combinations of features from the given data.
        
        Args:
            data: DataFrame containing the data
            features: Features to use for state generation. If None, will use
                     Gender, Type, Age, and Tenure.
                     
        Returns:
            List of state tuples
        """
        if features is None:
            features = ['Gender', 'Type', 'Age', 'Tenure']
            
        # Create all possible combinations of the unique values for each feature
        feature_values = [data[feature].unique() for feature in features]
        states = list(product(*feature_values))
        
        # Create a mapping from state to index for faster lookup
        self.states = states
        self.state_map = {state: i for i, state in enumerate(states)}
        
        return states
    
    def map_state_to_index(self, state: Tuple) -> int:
        """
        Map a state to its index in the state list.
        
        Args:
            state: State tuple
            
        Returns:
            Index of the state in the state list
        """
        state_tuple = tuple(state)
        if state_tuple in self.state_map:
            return self.state_map[state_tuple]
        # Fallback to hash if state not in map
        return hash(state_tuple) % len(self.states) if self.states else 0
    
    def generate_actions(self, data: pd.DataFrame, action_column: str = 'SubLine_Sent') -> np.ndarray:
        """
        Generate actions based on unique values in the specified column from the given data.
        
        Args:
            data: DataFrame containing the data
            action_column: Column name containing the action values
            
        Returns:
            Array of unique action values
        """
        actions = data[action_column].unique()
        self.actions = actions
        return actions
    
    def get_reward(self, state: Tuple, data: pd.DataFrame, reward_column: str = 'Reward') -> float:
        """
        Get the reward for the given state from the data.
        
        Args:
            state: State tuple
            data: DataFrame containing the rewards
            reward_column: Column name containing the rewards
            
        Returns:
            Reward value for the state
        """
        state_index = self.map_state_to_index(state)
        if state_index < len(data):
            return data.iloc[state_index][reward_column]
        return 0.0  # Default reward if state index out of bounds
    
    def calculate_reward_for_training(self, state: Tuple, data: pd.DataFrame, reward_column: str = 'Reward') -> float:
        """
        Calculate reward for a given state during training.
        
        Args:
            state: State tuple
            data: DataFrame containing the rewards
            reward_column: Column name containing the rewards
            
        Returns:
            Reward value for the state
        """
        return self.get_reward(state, data, reward_column)
    
    def get_subject_line(self, state: Tuple, data: pd.DataFrame, subject_column: str = 'SubLine_Sent') -> Any:
        """
        Get the suggested subject line for the given state from the data.
        
        Args:
            state: State tuple
            data: DataFrame containing the subject lines
            subject_column: Column name containing the subject lines
            
        Returns:
            Subject line for the state
        """
        state_index = self.map_state_to_index(state)
        if state_index < len(data):
            return data.iloc[state_index][subject_column]
        return None  # Default if state index out of bounds


# For backwards compatibility
def generate_states(data: pd.DataFrame) -> List[Tuple]:
    """
    Generate states based on unique combinations of Gender, Type, Age, and Tenure from the given data.
    
    Args:
        data: DataFrame containing the data
        
    Returns:
        List of state tuples
    """
    handler = StateActionHandler()
    return handler.generate_states(data)

def generate_actions(data: pd.DataFrame) -> np.ndarray:
    """
    Generate actions based on unique SubLine_Sent values from the given data.
    
    Args:
        data: DataFrame containing the data
        
    Returns:
        Array of unique action values
    """
    handler = StateActionHandler()
    return handler.generate_actions(data)

def get_reward(state: Tuple, data: pd.DataFrame, states: List[Tuple]) -> float:
    """
    Get the reward for the given state from the data.
    
    Args:
        state: State tuple
        data: DataFrame containing the rewards
        states: List of all possible states
        
    Returns:
        Reward value for the state
    """
    state_index = hash(tuple(state)) % len(states)
    if state_index < len(data):
        return data.iloc[state_index]['Reward']
    return 0.0 