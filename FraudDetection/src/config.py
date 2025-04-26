# src/config.py
import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config file. If None, use the default path.
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config',
            'config.yaml'
        )
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config