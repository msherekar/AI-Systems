#!/usr/bin/env python3
"""
Main application for Automated License Plate Recognition system.
This script orchestrates the components of the ALPR system.
"""

import logging
import argparse
import yaml
from pathlib import Path
import time
import os
import sys

# Add parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.video_processor import VideoProcessor
from src.detection.detector import LicensePlateDetector
from src.detection.text_recognizer import LicensePlateTextRecognizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('alpr.log')
    ]
)
logger = logging.getLogger('ALPR')

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration parameters.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def process_video_stream(config_path):
    """
    Process a video stream to detect and recognize license plates.
    
    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize components
    video_processor = VideoProcessor(config_path)
    detector = LicensePlateDetector(config_path)
    recognizer = LicensePlateTextRecognizer(config_path)
    
    # Initialize models
    detector.initialize()
    
    # Get video parameters from config
    input_url = config['video']['input_url']
    width = config['video']['width']
    height = config['video']['height']
    
    # Get output file path from config
    output_file = config['output']['results_file']
    
    logger.info(f"Starting ALPR system on stream: {input_url}")
    
    # Process video stream
    frames = video_processor.stream_to_frame(input_url, width, height)
    
    plate_count = {}
    frames_processed = 0
    start_time = time.time()
    
    try:
        for frame in frames:
            # Detect license plate
            license_plate_image = detector.detect_license_plate(frame)
            
            # If license plate detected, recognize text
            if license_plate_image is not None:
                plate_number = recognizer.read_license_plate(license_plate_image)
                
                # Count valid plate numbers
                if recognizer.is_valid_plate(plate_number):
                    plate_count[plate_number] = plate_count.get(plate_number, 0) + 1
                    logger.info(f"Detected plate: {plate_number}")
                    
            frames_processed += 1
            
            # Log progress every 100 frames
            if frames_processed % 100 == 0:
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frames_processed} frames ({fps:.2f} fps)")
                
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        
    except Exception as e:
        logger.error(f"Error processing video stream: {e}")
        
    finally:
        # Generate the final report
        logger.info(f"Number of unique plate numbers: {len(plate_count)}")
        for plate_number, count in plate_count.items():
            logger.info(f"Plate number {plate_number} detected {count} times.")
            
        # Save results to file
        recognizer.generate_report(plate_count, output_file)
        
        # Print completion message
        elapsed = time.time() - start_time
        fps = frames_processed / elapsed if elapsed > 0 else 0
        logger.info(f"Processing completed. Processed {frames_processed} frames in {elapsed:.2f} seconds ({fps:.2f} fps)")
        logger.info(f"Results saved to {output_file}")

def main():
    """
    Main entry point for the application.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Automated License Plate Recognition System')
    parser.add_argument('--config', default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', choices=['stream', 'api'], default='stream', help='Operation mode')
    parser.add_argument('--host', help='API host address')
    parser.add_argument('--port', type=int, help='API port')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Make path relative to script location
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / config_path
        
    logger.info(f"Using configuration from: {config_path}")
    
    if args.mode == 'stream':
        # Process video stream
        process_video_stream(config_path)
        
    elif args.mode == 'api':
        # Import API module only when needed
        from src.api.service import ALPRService
        
        # Start API service
        service = ALPRService(config_path)
        service.run(
            host=args.host, 
            port=args.port, 
            debug=args.debug
        )

if __name__ == "__main__":
    main() 