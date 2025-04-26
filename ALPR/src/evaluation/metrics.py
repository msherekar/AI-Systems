import os
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Metrics')

class Metrics:
    """
    Class for computing various evaluation metrics for object detection.
    """
    
    def __init__(self):
        """
        Initialize the Metrics class.
        """
        pass
        
    def jaccard_index(self, y_prediction, y_label):
        """
        Calculate the Jaccard Index (IoU) between predicted and ground truth bounding boxes or masks.

        Args:
            y_prediction (numpy.ndarray): Predicted bounding boxes or masks.
            y_label (numpy.ndarray): Ground truth bounding boxes or masks.

        Returns:
            float: Jaccard Index (IoU) score.
        """
        intersection = np.sum(np.logical_and(y_prediction, y_label))
        union = np.sum(np.logical_or(y_prediction, y_label))
        if union == 0:
            return 0  # To handle division by zero
        else:
            return intersection / union
            
    def calculate_precision(self, tp, fp):
        """
        Calculate precision.
        
        Args:
            tp (int): Number of true positives.
            fp (int): Number of false positives.
            
        Returns:
            float: Precision score.
        """
        return tp / (tp + fp) if (tp + fp) > 0 else 0
        
    def calculate_recall(self, tp, fn):
        """
        Calculate recall.
        
        Args:
            tp (int): Number of true positives.
            fn (int): Number of false negatives.
            
        Returns:
            float: Recall score.
        """
        return tp / (tp + fn) if (tp + fn) > 0 else 0
        
    def calculate_f1_score(self, precision, recall):
        """
        Calculate F1 score.
        
        Args:
            precision (float): Precision score.
            recall (float): Recall score.
            
        Returns:
            float: F1 score.
        """
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    def iou_bbox(self, box1, box2):
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            box1 (list): First bounding box [x, y, w, h].
            box2 (list): Second bounding box [x, y, w, h].
            
        Returns:
            float: IoU score.
        """
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection area
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def calculate_map(self, y_true, y_pred, iou_threshold=0.5):
        """
        Calculate mean Average Precision (mAP) at specified IoU threshold.
        
        Args:
            y_true (list): List of ground truth bounding boxes.
            y_pred (list): List of predicted bounding boxes with confidences.
            iou_threshold (float): IoU threshold for positive detection.
            
        Returns:
            float: mAP score.
        """
        # Simplified mAP calculation for demonstration
        # In a real implementation, this would be more complex with precision-recall curves
        
        if not y_true or not y_pred:
            return 0.0
            
        # Sort predictions by confidence (descending)
        y_pred.sort(key=lambda x: x['confidence'], reverse=True)
        
        tp = 0
        fp = 0
        
        # Count ground truth boxes
        n_gt = len(y_true)
        
        # Create array to track which ground truth boxes have been matched
        gt_matched = [False] * n_gt
        
        for pred in y_pred:
            pred_box = pred['box']
            match_found = False
            
            for i, gt_box in enumerate(y_true):
                if not gt_matched[i]:
                    iou = self.iou_bbox(pred_box, gt_box)
                    if iou >= iou_threshold:
                        tp += 1
                        gt_matched[i] = True
                        match_found = True
                        break
                        
            if not match_found:
                fp += 1
                
        # Calculate precision and recall
        precision = self.calculate_precision(tp, fp)
        recall = self.calculate_recall(tp, n_gt - tp)
        
        return precision  # Simplified mAP calculation
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate various metrics for object detection evaluation.
        
        Args:
            y_true (list): Ground truth annotations.
            y_pred (list): Predicted annotations.
            
        Returns:
            dict: Dictionary of computed metrics.
        """
        # Calculate various metrics
        mAP50 = self.calculate_map(y_true, y_pred, iou_threshold=0.5)
        
        # Calculate average IoU
        iou_values = []
        for gt, pred in zip(y_true, y_pred):
            iou = self.iou_bbox(gt, pred['box'])
            iou_values.append(iou)
            
        avg_iou = np.mean(iou_values) if len(iou_values) > 0 else 0
        
        # Create the metrics dictionary
        metrics = {
            'mAP50': mAP50,
            'average_iou': avg_iou
        }
        
        logger.info(f"Calculated metrics: {metrics}")
        return metrics
    
    def generate_report(self, y_prediction, y_label, report_filename):
        """
        Generate a report containing various metrics and store it in the 'results' directory.

        Args:
            y_prediction (numpy.ndarray): Predicted bounding boxes or masks.
            y_label (numpy.ndarray): Ground truth bounding boxes or masks.
            report_filename (str): Name of the report file.
        """
        # Ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_label, y_prediction)
        
        # Open the report file in write mode
        report_path = results_dir / report_filename
        try:
            with open(report_path, "w") as f:
                f.write("Object Detection Evaluation Metrics\n")
                f.write("==================================\n\n")
                
                for metric_name, metric_value in metrics.items():
                    f.write(f"{metric_name}: {metric_value:.4f}\n")
                    
            logger.info(f"Metrics report generated at {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating metrics report: {e}")
            
        return metrics
