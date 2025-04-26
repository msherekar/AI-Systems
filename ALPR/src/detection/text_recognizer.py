import logging
import yaml
from pathlib import Path

from ..models.ocr_model import OCRModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TextRecognizer')

class LicensePlateTextRecognizer:
    """
    Text recognizer class for reading text from license plate images using OCR.
    """
    
    def __init__(self, config_path='../../config/config.yaml'):
        """
        Initialize the text recognizer.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.ocr_model = OCRModel(config_path)
        
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
        Initialize the OCR model.
        
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        return self.ocr_model.load_model()
        
    def read_license_plate(self, license_plate_image):
        """
        Read text from a license plate image.
        
        Args:
            license_plate_image (numpy.ndarray): Cropped license plate image.
            
        Returns:
            str: Recognized license plate text or error message.
        """
        if license_plate_image is None:
            logger.warning("No license plate image provided for OCR")
            return "Nothing Detected"
            
        try:
            # Process the license plate image with OCR
            plate_text = self.ocr_model.predict(license_plate_image)
            
            logger.info(f"License plate text recognized: {plate_text}")
            return plate_text
            
        except Exception as e:
            logger.error(f"Error in license plate text recognition: {e}")
            return "OCR Error"
    
    def is_valid_plate(self, plate_text):
        """
        Check if the recognized plate text is valid.
        
        Args:
            plate_text (str): Recognized license plate text.
            
        Returns:
            bool: True if the plate text is valid, False otherwise.
        """
        return (plate_text not in ["Invalid Characters", "Invalid Length", "Nothing Detected", "OCR Error"])
    
    def process_results(self, license_plates, detected_texts=None):
        """
        Process a batch of license plate detections and recognitions.
        
        Args:
            license_plates (list): List of license plate images.
            detected_texts (list): Optional list of pre-recognized texts.
            
        Returns:
            dict: Dictionary with plate texts and their counts.
        """
        plate_counts = {}
        
        if detected_texts is None:
            detected_texts = []
            for plate_image in license_plates:
                if plate_image is not None:
                    text = self.read_license_plate(plate_image)
                    detected_texts.append(text)
        
        # Count occurrences of each valid plate
        for text in detected_texts:
            if self.is_valid_plate(text):
                plate_counts[text] = plate_counts.get(text, 0) + 1
                
        return plate_counts
    
    def generate_report(self, plate_counts, output_file=None):
        """
        Generate a report of detected license plates.
        
        Args:
            plate_counts (dict): Dictionary with plate texts and their counts.
            output_file (str): Path to output file.
            
        Returns:
            str: Report text.
        """
        # Generate the report text
        report_text = f"Number of unique plate numbers: {len(plate_counts)}\n\n"
        
        for plate_number, count in plate_counts.items():
            report_text += f"Plate number {plate_number} detected {count} times.\n"
            
        report_text += "\nThe video has been done analysing.\n"
        
        # If output file specified, write to file
        if output_file is not None:
            try:
                with open(output_file, "w") as f:
                    f.write(report_text)
                logger.info(f"Results saved to {output_file}")
            except Exception as e:
                logger.error(f"Error writing results to file: {e}")
        
        return report_text
