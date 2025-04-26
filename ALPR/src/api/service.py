from flask import Flask, request, jsonify
import yaml
import logging
import cv2 as cv
import numpy as np
import base64
from pathlib import Path
import os
import time

from ..detection.detector import LicensePlateDetector
from ..detection.text_recognizer import LicensePlateTextRecognizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('APIService')

class ALPRService:
    """
    REST API service for Automated License Plate Recognition.
    """
    
    def __init__(self, config_path='../../config/config.yaml'):
        """
        Initialize the API service.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.app = Flask(__name__)
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize the detector and recognizer
        self.detector = LicensePlateDetector(config_path)
        self.recognizer = LicensePlateTextRecognizer(config_path)
        
        # Initialize models
        self.detector.initialize()
        self.recognizer.initialize()
        
        # Set up routes
        self.setup_routes()
        
    def _load_config(self):
        """
        Load configuration from YAML file.
        
        Returns:
            dict: Configuration parameters.
        """
        try:
            config_path = Path(self.config_path)
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def setup_routes(self):
        """
        Set up the API routes.
        """
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """
            Health check endpoint.
            """
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time()
            })
        
        @self.app.route('/detect', methods=['POST'])
        def detect_license_plate():
            """
            Detect license plate in an image.
            """
            try:
                # Get request data
                data = request.json
                
                if not data or 'image' not in data:
                    return jsonify({
                        'error': 'No image data provided'
                    }), 400
                
                # Decode base64 image
                image_data = data['image']
                draw_result = data.get('draw_result', False)
                
                # Convert base64 to image
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv.imdecode(nparr, cv.IMREAD_COLOR)
                
                # Process the image
                license_plate, annotated_image = self.detector.process_image(img, draw_result)
                
                # If no license plate detected
                if license_plate is None:
                    return jsonify({
                        'success': False,
                        'message': 'No license plate detected in the image'
                    })
                
                # Recognize text from license plate
                plate_text = self.recognizer.read_license_plate(license_plate)
                
                # Prepare response
                response = {
                    'success': True,
                    'plate_text': plate_text,
                    'valid': self.recognizer.is_valid_plate(plate_text)
                }
                
                # Include annotated image if requested
                if draw_result and annotated_image is not None:
                    _, buffer = cv.imencode('.jpg', annotated_image)
                    annotated_base64 = base64.b64encode(buffer).decode('utf-8')
                    response['annotated_image'] = annotated_base64
                    
                # Include cropped license plate image
                _, buffer = cv.imencode('.jpg', license_plate)
                plate_base64 = base64.b64encode(buffer).decode('utf-8')
                response['license_plate_image'] = plate_base64
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error in license plate detection: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/batch-process', methods=['POST'])
        def batch_process():
            """
            Process multiple images in batch.
            """
            try:
                # Get request data
                data = request.json
                
                if not data or 'images' not in data:
                    return jsonify({
                        'error': 'No images provided'
                    }), 400
                
                images_data = data['images']
                results = []
                
                for image_data in images_data:
                    # Convert base64 to image
                    img_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
                    
                    # Process the image
                    license_plate, _ = self.detector.process_image(img, False)
                    
                    if license_plate is None:
                        results.append({
                            'success': False,
                            'message': 'No license plate detected'
                        })
                        continue
                    
                    # Recognize text from license plate
                    plate_text = self.recognizer.read_license_plate(license_plate)
                    
                    results.append({
                        'success': True,
                        'plate_text': plate_text,
                        'valid': self.recognizer.is_valid_plate(plate_text)
                    })
                
                return jsonify({
                    'success': True,
                    'results': results
                })
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def run(self, host=None, port=None, debug=False):
        """
        Run the Flask API service.
        
        Args:
            host (str): Host address to run the service on.
            port (int): Port to run the service on.
            debug (bool): Whether to run in debug mode.
        """
        if host is None:
            host = self.config['api']['host']
        if port is None:
            port = self.config['api']['port']
            
        logger.info(f"Starting ALPR API service on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_app(config_path='../../config/config.yaml'):
    """
    Create and configure the Flask application.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        Flask: The configured Flask application.
    """
    service = ALPRService(config_path)
    return service.app


if __name__ == '__main__':
    # When running as a script, start the service
    service = ALPRService()
    service.run(debug=True)
