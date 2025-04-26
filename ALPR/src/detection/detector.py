import logging
import yaml
from pathlib import Path
import numpy as np
import cv2 as cv

from ..models.yolo_model import YOLOModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PlateDetector')

class LicensePlateDetector:
    """
    Detector class for finding license plates in images.
    """
    
    def __init__(self, config_path='../../config/config.yaml', model_type='yolo_tiny'):
        """
        Initialize the license plate detector.
        
        Args:
            config_path (str): Path to the configuration file.
            model_type (str): Type of detection model to use.
        """
        self.config_path = config_path
        self.model_type = model_type
        self.config = self._load_config()
        self.model = YOLOModel(config_path, model_type)
        
    def _load_config(self):
        """
        Load configuration from YAML file.
        
        Returns:
            dict: Configuration parameters.
        """
        config_path = Path(self.config_path)
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def initialize(self):
        """
        Initialize the detector by loading the model.
        
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        return self.model.load_model()
        
    def detect_license_plate(self, frame):
        """
        Detect license plate in a given frame.
        
        Args:
            frame (numpy.ndarray): Input image frame.
            
        Returns:
            numpy.ndarray: Cropped license plate image or None if not detected.
        """
        try:
            # Get detection results from model
            detection_results = self.model.predict(frame)
            
            # If no detections, return None
            if len(detection_results['boxes']) == 0:
                return None
                
            # Apply NMS to filter detections
            boxes, confidences, class_ids = self.model.apply_nms(detection_results)
            
            # If no boxes after NMS, return None
            if len(boxes) == 0:
                return None
                
            # Take the first detected license plate (highest confidence after NMS)
            x, y, w, h = boxes[0]
            
            # Crop the license plate from the frame
            license_plate = frame[y:y+h, x:x+w]
            
            logger.info(f"License plate detected with confidence {confidences[0]:.2f}")
            
            return license_plate
            
        except Exception as e:
            logger.error(f"Error in license plate detection: {e}")
            return None
    
    def draw_detection(self, frame, detections):
        """
        Draw detection boxes on the frame.
        
        Args:
            frame (numpy.ndarray): Input image frame.
            detections (tuple): Boxes, confidences, and class IDs.
            
        Returns:
            numpy.ndarray: Frame with detection boxes drawn.
        """
        boxes, confidences, class_ids = detections
        
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Draw bounding boxes
        for i, box in enumerate(boxes):
            x, y, w, h = box
            confidence = confidences[i]
            
            # Draw rectangle
            cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add confidence text
            text = f"License Plate: {confidence:.2f}"
            cv.putText(output_frame, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return output_frame
    
    def process_image(self, image, draw_result=False):
        """
        Process an image to detect license plates.
        
        Args:
            image (numpy.ndarray): Input image.
            draw_result (bool): Whether to draw detection results on the image.
            
        Returns:
            tuple: Cropped license plate and optionally the annotated image.
        """
        # Detect license plate
        detection_results = self.model.predict(image)
        
        if len(detection_results['boxes']) == 0:
            logger.info("No license plate detected")
            return None, image if draw_result else None
            
        # Apply NMS
        detections = self.model.apply_nms(detection_results)
        boxes, confidences, class_ids = detections
        
        if len(boxes) == 0:
            logger.info("No license plate detected after NMS")
            return None, image if draw_result else None
            
        # Crop the license plate
        x, y, w, h = boxes[0]
        license_plate = image[y:y+h, x:x+w]
        
        # Draw detections if requested
        annotated_image = None
        if draw_result:
            annotated_image = self.draw_detection(image, detections)
            
        return license_plate, annotated_image
