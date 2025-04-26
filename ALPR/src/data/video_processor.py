import ffmpeg
import numpy as np
import yaml
import cv2 as cv
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VideoProcessor')

class VideoProcessor:
    """
    Handles video processing tasks including frame extraction and basic preprocessing.
    """
    
    def __init__(self, config_path='../../config/config.yaml'):
        """
        Initialize the video processor with configuration.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """
        Load configuration from YAML file.
        
        Returns:
            dict: Configuration parameters.
        """
        config_path = Path(self.config_path)
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def stream_to_frame(self, input_url=None, width=None, height=None):
        """
        Convert video stream to individual frames.
        
        Args:
            input_url (str): URL of the video stream.
            width (int): Width of the video frames.
            height (int): Height of the video frames.
            
        Yields:
            numpy.ndarray: Video frame.
        """
        # Use provided parameters or get from config
        if input_url is None:
            input_url = self.config['video']['input_url']
        if width is None:
            width = self.config['video']['width']
        if height is None:
            height = self.config['video']['height']
            
        logger.info(f"Starting video stream from {input_url}")
        
        try:
            process = (
                ffmpeg
                .input(input_url)
                .output('pipe:', format='rawvideo', pix_fmt='bgr24')
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            
            frame_count = 0
            while True:
                # Read raw bytes from the video stream
                in_bytes = process.stdout.read(width * height * 3)
                
                # Break loop if no more frames
                if not in_bytes:
                    break
                    
                # Convert bytes to numpy array
                in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
                    
                yield in_frame  # Yield the frame
                
        except Exception as e:
            logger.error(f"Error processing video stream: {e}")
    
    def video_to_frames(self, video_path, max_frames=None):
        """
        Convert video file to individual frames.
        
        Args:
            video_path (str): Path to the video file.
            max_frames (int): Maximum number of frames to extract.
            
        Returns:
            list: List of frames as numpy arrays.
        """
        logger.info(f"Reading video from file: {video_path}")
        frames = []
        
        cap = cv.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret or (max_frames is not None and frame_count >= max_frames):
                break
                
            frames.append(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Extracted {frame_count} frames from video")
                
        cap.release()
        logger.info(f"Total frames extracted: {len(frames)}")
        return frames
    
    def resize_frame(self, frame, width=None, height=None):
        """
        Resize a frame to specified dimensions.
        
        Args:
            frame (numpy.ndarray): Input frame.
            width (int): New width.
            height (int): New height.
            
        Returns:
            numpy.ndarray: Resized frame.
        """
        if width is None or height is None:
            return frame
            
        return cv.resize(frame, (width, height))
    
    def save_frame(self, frame, output_path):
        """
        Save a frame to disk.
        
        Args:
            frame (numpy.ndarray): Frame to save.
            output_path (str): Path to save the frame.
            
        Returns:
            bool: True if saved successfully, False otherwise.
        """
        try:
            cv.imwrite(output_path, frame)
            return True
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False
