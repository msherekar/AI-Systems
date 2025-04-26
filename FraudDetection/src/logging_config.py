# src/logging_config.py
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional

def setup_logging(config: Optional[Dict[str, Any]] = None, log_file: Optional[str] = None):
    """
    Configure logging for the application.
    
    Args:
        config: Configuration dictionary
        log_file: Path to the log file. If None, use stderr.
    """
    if config is None:
        config = {}
        
    log_level = config.get('log_level', 'INFO').upper()
    log_format = config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)