import unittest
import sys
import os
import numpy as np
import cv2 as cv
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import LicensePlateDetector

class TestLicensePlateDetector(unittest.TestCase):
    """
    Test suite for LicensePlateDetector class.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Use a test config file
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "config", "config.yaml")
        self.detector = LicensePlateDetector(self.config_path)
        self.detector.initialize()
        
        # Create a test image (black background with white rectangle)
        self.test_image = np.zeros((300, 500, 3), dtype=np.uint8)
        # Draw a white rectangle to simulate a license plate
        cv.rectangle(self.test_image, (150, 100), (350, 200), (255, 255, 255), -1)
        
    def test_initialization(self):
        """
        Test detector initialization.
        """
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.model)
        
    def test_process_image(self):
        """
        Test image processing functionality.
        Note: This test will not detect actual license plates since we're using a dummy image,
        but it tests that the code runs without errors.
        """
        # Process the test image
        license_plate, annotated_image = self.detector.process_image(self.test_image, True)
        
        # The test image doesn't have a real license plate, so result should be None
        # but the function should complete without errors
        self.assertIsNone(license_plate)
        self.assertIsNotNone(annotated_image)
        
    def test_draw_detection(self):
        """
        Test detection box drawing functionality.
        """
        # Create mock detection results
        boxes = [[150, 100, 200, 100]]  # [x, y, w, h]
        confidences = [0.95]
        class_ids = [0]
        
        detections = (boxes, confidences, class_ids)
        
        # Draw detections on the test image
        result = self.detector.draw_detection(self.test_image, detections)
        
        # Ensure result is not None and has the same shape as the input
        self.assertIsNotNone(result)
        self.assertEqual(self.test_image.shape, result.shape)
        
        # The result should be different from the original image (because we drew on it)
        self.assertFalse(np.array_equal(self.test_image, result))

if __name__ == '__main__':
    unittest.main()
