# src/api/service.py
from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import traceback
import tempfile

from ..models.model_factory import get_model
from ..data.etl import ETLPipeline
from ..config import load_config

logger = logging.getLogger(__name__)

class FraudDetectionService:
    """Fraud detection API service."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the fraud detection service.
        
        Args:
            config: Configuration dictionary. If None, load from default path.
        """
        if config is None:
            self.config = load_config()
        else:
            self.config = config
            
        self.app = Flask(__name__)
        self.model_type = self.config.get('api', {}).get('default_model', 'random_forest')
        self.model = get_model(self.model_type, self.config)
        self.etl = ETLPipeline(self.config)
        
        # Set up routes
        self.setup_routes()
        
    def setup_routes(self):
        """Set up API routes."""
        self.app.route('/health', methods=['GET'])(self.health)
        self.app.route('/detect', methods=['POST'])(self.detect_fraud_post)
        self.app.route('/infer', methods=['GET'])(self.detect_fraud_get)
        self.app.errorhandler(Exception)(self.handle_error)
        
    def health(self):
        """Health check endpoint."""
        return jsonify({
            'status': 'ok',
            'model': self.model_type,
            'version': '1.0.0'
        })
        
    def process_transaction_data(self, data: pd.DataFrame) -> str:
        """
        Process transaction data and make a fraud prediction.
        
        Args:
            data: DataFrame containing transaction data
            
        Returns:
            Fraud determination message
        """
        try:
            # Transform the data
            transformed_data = self.etl.transform(data)
            
            # Make prediction
            prediction = self.model.predict(transformed_data)
            
            # Determine result
            if prediction[0] == 1:
                determination = 'FRAUD: This transaction appears to be fraudulent'
            else:
                determination = 'LEGITIMATE: This transaction appears to be legitimate'
                
            return determination
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            logger.error(traceback.format_exc())
            return f"ERROR: Could not process transaction data: {str(e)}"
        
    def detect_fraud_post(self):
        """POST endpoint for fraud detection."""
        try:
            # Get the file from the request
            transaction_file = request.files.get('input_transaction')
            if not transaction_file:
                return jsonify({
                    'status': 'error',
                    'message': 'No transaction file provided'
                }), 400
                
            # Save the file to a temporary location
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
                transaction_file.save(temp.name)
                temp_path = temp.name
                
            # Load and process the data
            data = pd.read_csv(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Process the data
            result = self.process_transaction_data(data)
            
            return jsonify({
                'status': 'success',
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Error in POST endpoint: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
    def detect_fraud_get(self):
        """GET endpoint for fraud detection."""
        try:
            # Get the file path from query parameter
            csv_file = request.args.get('check_transaction')
            if not csv_file:
                return jsonify({
                    'status': 'error',
                    'message': 'No transaction file specified'
                }), 400
                
            # Check if the file exists
            if not os.path.exists(csv_file):
                return jsonify({
                    'status': 'error',
                    'message': f'File not found: {csv_file}'
                }), 404
                
            # Load and process the data
            data = pd.read_csv(csv_file)
            
            # Process the data
            result = self.process_transaction_data(data)
            
            return jsonify({
                'status': 'success',
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Error in GET endpoint: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
    def handle_error(self, e):
        """Global error handler."""
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'An internal server error occurred'
        }), 500
        
    def run(self, host=None, port=None, debug=None):
        """
        Run the Flask application.
        
        Args:
            host: Host to run the server on. If None, use config value.
            port: Port to run the server on. If None, use config value.
            debug: Whether to run in debug mode. If None, use config value.
        """
        host = host or self.config.get('api', {}).get('host', '0.0.0.0')
        port = port or self.config.get('api', {}).get('port', 8786)
        debug = debug if debug is not None else self.config.get('api', {}).get('debug', False)
        
        logger.info(f"Starting Fraud Detection Service on {host}:{port} (debug={debug})")
        self.app.run(host=host, port=port, debug=debug)