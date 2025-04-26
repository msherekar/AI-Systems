#!/usr/bin/env python3
"""
Training script for the reinforcement learning model.

This script trains the Q-learning agent using the provided configuration
and data files.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import Trainer
from src.training.reporting import Reporter


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config):
    """Set up logging based on configuration."""
    logging_config = config.get('logging', {})
    log_level = getattr(logging, logging_config.get('level', 'INFO'))
    log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = logging_config.get('file', None)
    
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
            logging.StreamHandler()
        ]
    )


def main():
    """Train the Q-learning agent."""
    parser = argparse.ArgumentParser(description='Train the Q-learning agent')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--userbase_file', type=str, help='Path to userbase CSV file (overrides config)')
    parser.add_argument('--sent_file', type=str, help='Path to sent emails CSV file (overrides config)')
    parser.add_argument('--responded_file', type=str, help='Path to responded emails CSV file (overrides config)')
    parser.add_argument('--episodes', type=int, help='Number of training episodes (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info('Starting training script')
    
    # Get data file paths
    userbase_file = args.userbase_file or config['data']['userbase_file']
    sent_file = args.sent_file or config['data']['sent_emails_file']
    responded_file = args.responded_file or config['data']['responded_emails_file']
    
    # Get training parameters
    episodes = args.episodes or config['training']['episodes']
    agent_params = config['model']
    output_dir = config['model']['output_dir']
    metrics_dir = config['training']['metrics_dir']
    logs_dir = config['training']['logs_dir']
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Initialize trainer
    logger.info('Initializing trainer')
    trainer = Trainer(
        agent_params=agent_params,
        output_dir=output_dir,
        metrics_dir=metrics_dir,
        logs_dir=logs_dir
    )
    
    # Preprocess data
    logger.info('Preprocessing data')
    processed_data = trainer.preprocess_data(userbase_file, sent_file, responded_file)
    
    # Save processed data if configured
    processed_file = config['data']['processed_file']
    if processed_file:
        logger.info(f'Saving processed data to {processed_file}')
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        processed_data.to_csv(processed_file, index=False)
    
    # Train the model
    logger.info(f'Starting training for {episodes} episodes')
    training_results = trainer.train(processed_data, episodes=episodes)
    
    # Generate report
    logger.info('Generating training report')
    reporter = Reporter(output_dir=metrics_dir)
    report_dir = reporter.generate_report(training_results)
    
    logger.info(f'Training completed. Report saved to {report_dir}')
    logger.info(f'Model saved to {os.path.join(output_dir, "q_table.pkl")}')


if __name__ == '__main__':
    main() 