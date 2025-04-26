# Configuration manager

import os
import yaml
from pathlib import Path

class ConfigManager:
    """
    Class to manage configuration parameters for the TimeSeries project.
    """

    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize the config manager.

        Args:
            config_path (str): Path to the config file.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """
        Load the configuration from the YAML file.

        Returns:
            dict: Configuration parameters.
        """
        # Get the absolute path to the config file
        base_path = Path(__file__).parent.parent.parent
        config_file = base_path / self.config_path

        # Check if the config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # Load the configuration
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_data_paths(self):
        """
        Get the data paths from the configuration.

        Returns:
            dict: Data paths.
        """
        return self.config.get('data', {})

    def get_model_params(self, model_type, model_name):
        """
        Get the parameters for a specific model.

        Args:
            model_type (str): Type of the model (sarimax or lstm).
            model_name (str): Name of the model (total or fraud).

        Returns:
            dict: Model parameters.
        """
        return self.config.get('models', {}).get(model_type, {}).get(model_name, {})

    def get_training_params(self):
        """
        Get the training parameters from the configuration.

        Returns:
            dict: Training parameters.
        """
        return self.config.get('training', {})

    def get_api_settings(self):
        """
        Get the API settings from the configuration.

        Returns:
            dict: API settings.
        """
        return self.config.get('api', {}) 