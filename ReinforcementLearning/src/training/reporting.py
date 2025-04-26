"""
Training report generation utilities.

This module contains the Reporter class for generating comprehensive
reports on the training results.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Reporter:
    """
    Reporter class for generating training reports.
    
    This class provides methods to generate comprehensive reports on the
    training results, including metrics, visualizations, and analysis.
    """
    
    def __init__(self, output_dir: str = 'reports'):
        """
        Initialize the Reporter.
        
        Args:
            output_dir: Directory to save the reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_metrics_to_json(self, metrics: Dict[str, Any], filename: str) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics
            filename: Output filename
        """
        # Convert numpy arrays and lists to regular lists for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                json_metrics[key] = [v.tolist() for v in value]
            else:
                json_metrics[key] = value
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(json_metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {filepath}")
        
    def save_metrics_to_csv(self, metrics: Dict[str, List[float]], filename: str) -> None:
        """
        Save metrics to a CSV file.
        
        Args:
            metrics: Dictionary of metrics where values are lists of floats
            filename: Output filename
        """
        # Filter metrics to only include lists of floats of the same length
        csv_metrics = {}
        max_len = 0
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                csv_metrics[key] = value
                max_len = max(max_len, len(value))
        
        # Create DataFrame with metrics as columns
        df = pd.DataFrame()
        for key, value in csv_metrics.items():
            # Pad shorter lists with NaN
            if len(value) < max_len:
                value = value + [np.nan] * (max_len - len(value))
            df[key] = value
        
        # Add episode column
        df['episode'] = list(range(1, max_len + 1))
        
        # Reorder columns to have episode first
        cols = df.columns.tolist()
        cols = ['episode'] + [c for c in cols if c != 'episode']
        df = df[cols]
        
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Metrics saved to {filepath}")
        
    def plot_reward_comparison(self, rewards_per_episode: List[float], baseline_rewards: List[float] = None, 
                             title: str = 'Reward Comparison', filename: str = 'reward_comparison.png') -> None:
        """
        Plot a comparison of rewards with an optional baseline.
        
        Args:
            rewards_per_episode: List of rewards for each episode
            baseline_rewards: List of baseline rewards for comparison
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(12, 6))
        
        # Plot actual rewards
        plt.plot(rewards_per_episode, label='RL Agent', color='blue')
        
        # Plot baseline if provided
        if baseline_rewards is not None:
            plt.plot(baseline_rewards, label='Baseline', color='red', linestyle='--')
            
        # Calculate improvement
        if baseline_rewards is not None and len(baseline_rewards) > 0:
            avg_baseline = np.mean(baseline_rewards)
            avg_agent = np.mean(rewards_per_episode[-100:])  # Last 100 episodes
            improvement = (avg_agent - avg_baseline) / avg_baseline * 100 if avg_baseline != 0 else float('inf')
            plt.title(f"{title} (Improvement: {improvement:.2f}%)")
        else:
            plt.title(title)
            
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.legend()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        
        logger.info(f"Reward comparison plot saved to {filepath}")
        
    def generate_report(self, training_results: Dict[str, Any], baseline_rewards: List[float] = None) -> str:
        """
        Generate a comprehensive report on the training results.
        
        Args:
            training_results: Dictionary containing training results
            baseline_rewards: Optional baseline rewards for comparison
            
        Returns:
            Path to the generated report file
        """
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        report_dir = os.path.join(self.output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Extract metrics from training results
        metrics = training_results.get('metrics', {})
        rewards_per_episode = training_results.get('rewards_per_episode', [])
        
        # Save metrics to JSON and CSV
        self.save_metrics_to_json(metrics, os.path.join(report_dir, 'metrics.json'))
        
        # Extract metrics that are lists of floats
        metrics_for_csv = {
            'rewards_per_episode': rewards_per_episode,
            'cumulative_rewards': metrics.get('cumulative_rewards', []),
            'average_reward': metrics.get('average_reward', []),
            'discounted_rewards': metrics.get('discounted_rewards', [])
        }
        self.save_metrics_to_csv(metrics_for_csv, os.path.join(report_dir, 'metrics.csv'))
        
        # Plot reward comparison if baseline provided
        if baseline_rewards is not None:
            self.plot_reward_comparison(
                rewards_per_episode, 
                baseline_rewards, 
                filename=os.path.join(report_dir, 'reward_comparison.png')
            )
        
        # Generate summary report
        report_path = os.path.join(report_dir, 'summary.txt')
        with open(report_path, 'w') as f:
            f.write("# Training Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Training parameters
            f.write("## Training Parameters\n\n")
            agent = training_results.get('agent')
            if agent:
                f.write(f"- Learning Rate: {agent.learning_rate}\n")
                f.write(f"- Discount Factor: {agent.discount_factor}\n")
                f.write(f"- Epsilon: {agent.epsilon}\n")
            
            # Training results
            f.write("\n## Training Results\n\n")
            f.write(f"- Training Time: {training_results.get('training_time', 0):.2f} seconds\n")
            f.write(f"- Episodes to Convergence: {metrics.get('episodes_to_convergence', 'N/A')}\n")
            
            if len(rewards_per_episode) > 0:
                f.write(f"- Initial Average Reward: {rewards_per_episode[0]:.4f}\n")
                f.write(f"- Final Average Reward: {rewards_per_episode[-1]:.4f}\n")
                
                # Calculate improvement
                if len(rewards_per_episode) > 100:
                    initial_avg = np.mean(rewards_per_episode[:100])
                    final_avg = np.mean(rewards_per_episode[-100:])
                    improvement = (final_avg - initial_avg) / initial_avg * 100 if initial_avg != 0 else float('inf')
                    f.write(f"- Reward Improvement: {improvement:.2f}%\n")
            
            # Baseline comparison
            if baseline_rewards is not None and len(baseline_rewards) > 0:
                f.write("\n## Baseline Comparison\n\n")
                avg_baseline = np.mean(baseline_rewards)
                avg_agent = np.mean(rewards_per_episode[-100:])  # Last 100 episodes
                improvement = (avg_agent - avg_baseline) / avg_baseline * 100 if avg_baseline != 0 else float('inf')
                f.write(f"- Average Baseline Reward: {avg_baseline:.4f}\n")
                f.write(f"- Average Agent Reward: {avg_agent:.4f}\n")
                f.write(f"- Improvement over Baseline: {improvement:.2f}%\n")
            
            # Figures
            f.write("\n## Generated Figures\n\n")
            f.write("- [Cumulative Rewards](cumulative_rewards.png)\n")
            f.write("- [Average Reward](average_reward.png)\n")
            f.write("- [Discounted Rewards](discounted_rewards.png)\n")
            if baseline_rewards is not None:
                f.write("- [Reward Comparison](reward_comparison.png)\n")
        
        logger.info(f"Report generated at {report_dir}")
        return report_dir 