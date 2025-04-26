#!/usr/bin/env python3
"""
Deployment script for the reinforcement learning model.

This script starts the Flask API service that serves the trained model.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.service import run_app


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


def setup_environment(config):
    """Set up environment variables based on configuration."""
    # Set API configuration
    api_config = config.get('api', {})
    os.environ['PORT'] = str(api_config.get('port', 5000))
    os.environ['DEBUG'] = str(api_config.get('debug', False)).lower()
    
    # Set subject lines
    subject_lines = api_config.get('subject_lines', [])
    if subject_lines:
        os.environ['SUBJECT_LINES'] = ','.join(subject_lines)


def main():
    """Start the Flask API service."""
    parser = argparse.ArgumentParser(description='Start the Flask API service')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--host', type=str, help='Host to bind the server to (overrides config)')
    parser.add_argument('--port', type=int, help='Port to bind the server to (overrides config)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info('Starting deployment script')
    
    # Set up environment variables
    setup_environment(config)
    
    # Get API configuration
    api_config = config.get('api', {})
    host = args.host or api_config.get('host', '0.0.0.0')
    port = args.port or int(os.environ.get('PORT', 5000))
    debug = args.debug or (os.environ.get('DEBUG', 'false').lower() == 'true')
    
    # Start the Flask application
    logger.info(f'Starting Flask API service on {host}:{port}')
    run_app(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main() 