import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from .metrics import Metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AdvancedMetrics')

class AdvancedMetrics(Metrics):
    """
    Class for computing advanced evaluation metrics for object detection.
    Extends the base Metrics class with more sophisticated metrics and visualizations.
    """
    
    def __init__(self):
        """
        Initialize the AdvancedMetrics class.
        """
        super().__init__()
    
    def _calculate_ap_per_class(self, y_true, y_pred, iou_threshold=0.5, class_id=None):
        """
        Calculate Average Precision (AP) for a specific class.
        
        Args:
            y_true (list): List of ground truth bounding boxes.
            y_pred (list): List of predicted bounding boxes with confidences.
            iou_threshold (float): IoU threshold for positive detection.
            class_id (int): Class ID to calculate AP for.
            
        Returns:
            float: AP score for the specified class.
        """
        # Filter ground truth and predictions by class if specified
        if class_id is not None:
            y_true = [box for box in y_true if box.get('class_id') == class_id]
            y_pred = [box for box in y_pred if box.get('class_id') == class_id]
        
        if not y_true or not y_pred:
            return 0.0
        
        # Sort predictions by confidence (descending)
        y_pred = sorted(y_pred, key=lambda x: x['confidence'], reverse=True)
        
        # Initialize precision and recall arrays
        precision = []
        recall = []
        
        # Count ground truth boxes
        n_gt = len(y_true)
        
        # Create array to track which ground truth boxes have been matched
        gt_matched = [False] * n_gt
        
        tp = 0  # True positives
        fp = 0  # False positives
        
        # Iterate through predictions
        for pred in y_pred:
            pred_box = pred['box']
            match_found = False
            
            # Check for match with ground truth
            for i, gt in enumerate(y_true):
                if not gt_matched[i]:
                    gt_box = gt['box']
                    iou = self.iou_bbox(pred_box, gt_box)
                    
                    if iou >= iou_threshold:
                        tp += 1
                        gt_matched[i] = True
                        match_found = True
                        break
            
            if not match_found:
                fp += 1
            
            # Calculate current precision and recall
            current_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            current_recall = tp / n_gt if n_gt > 0 else 0
            
            precision.append(current_precision)
            recall.append(current_recall)
        
        # If no predictions, return 0 AP
        if not precision:
            return 0.0
        
        # Convert lists to numpy arrays
        precision = np.array(precision)
        recall = np.array(recall)
        
        # Ensure monotonically decreasing precision
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        
        # Find points where recall increases
        i = np.where(recall[1:] != recall[:-1])[0]
        
        # Calculate AP as area under precision-recall curve (using simple average)
        ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
        
        return ap
    
    def mAP(self, y_true, y_pred, iou_threshold=0.5, class_ids=None):
        """
        Calculate mean Average Precision (mAP) at specified IoU threshold across all classes.
        
        Args:
            y_true (list): List of ground truth bounding boxes.
            y_pred (list): List of predicted bounding boxes with confidences.
            iou_threshold (float): IoU threshold for positive detection.
            class_ids (list): List of class IDs to include in mAP calculation.
            
        Returns:
            float: mAP score.
        """
        # Get unique class IDs if not provided
        if class_ids is None:
            gt_classes = set(gt.get('class_id', 0) for gt in y_true)
            pred_classes = set(pred.get('class_id', 0) for pred in y_pred)
            class_ids = list(gt_classes.union(pred_classes))
        
        # Calculate AP for each class
        ap_per_class = {}
        for class_id in class_ids:
            ap = self._calculate_ap_per_class(y_true, y_pred, iou_threshold, class_id)
            ap_per_class[class_id] = ap
        
        # Calculate mAP
        mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
        
        return mAP
    
    def mAP50_95(self, y_true, y_pred, class_ids=None):
        """
        Calculate mean Average Precision (mAP) averaged over IoU thresholds from 0.5 to 0.95.
        
        Args:
            y_true (list): List of ground truth bounding boxes.
            y_pred (list): List of predicted bounding boxes with confidences.
            class_ids (list): List of class IDs to include in mAP calculation.
            
        Returns:
            float: mAP50-95 score.
        """
        thresholds = np.linspace(0.5, 0.95, 10)
        maps = []
        
        for threshold in thresholds:
            map_at_threshold = self.mAP(y_true, y_pred, threshold, class_ids)
            maps.append(map_at_threshold)
        
        mAP50_95 = np.mean(maps) if maps else 0.0
        
        return mAP50_95
    
    def calculate_advanced_metrics(self, y_true, y_pred):
        """
        Calculate advanced metrics for object detection evaluation.
        
        Args:
            y_true (list): Ground truth annotations.
            y_pred (list): Predicted annotations.
            
        Returns:
            dict: Dictionary of computed metrics.
        """
        # Calculate basic metrics from parent class
        basic_metrics = self.calculate_metrics(y_true, y_pred)
        
        # Calculate advanced metrics
        mAP50 = self.mAP(y_true, y_pred, 0.5)
        mAP_avg = self.mAP50_95(y_true, y_pred)
        
        # Add advanced metrics to dictionary
        advanced_metrics = {
            **basic_metrics,
            'mAP50_detailed': mAP50,
            'mAP50_95': mAP_avg
        }
        
        logger.info(f"Calculated advanced metrics: {advanced_metrics}")
        return advanced_metrics
    
    def plot_precision_recall_curve(self, y_true, y_pred, iou_threshold=0.5, save_path=None):
        """
        Plot precision-recall curve.
        
        Args:
            y_true (list): Ground truth annotations.
            y_pred (list): Predicted annotations.
            iou_threshold (float): IoU threshold for positive detection.
            save_path (str): Path to save the plot.
            
        Returns:
            tuple: Precision and recall arrays.
        """
        # Sort predictions by confidence
        y_pred = sorted(y_pred, key=lambda x: x['confidence'], reverse=True)
        
        n_gt = len(y_true)
        gt_matched = [False] * n_gt
        
        precision = []
        recall = []
        
        tp = 0
        fp = 0
        
        for pred in y_pred:
            pred_box = pred['box']
            match_found = False
            
            for i, gt in enumerate(y_true):
                if not gt_matched[i]:
                    gt_box = gt['box']
                    iou = self.iou_bbox(pred_box, gt_box)
                    
                    if iou >= iou_threshold:
                        tp += 1
                        gt_matched[i] = True
                        match_found = True
                        break
            
            if not match_found:
                fp += 1
            
            current_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            current_recall = tp / n_gt if n_gt > 0 else 0
            
            precision.append(current_precision)
            recall.append(current_recall)
        
        # Convert to numpy arrays
        precision = np.array(precision)
        recall = np.array(recall)
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, marker='.', label=f'IoU={iou_threshold}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        return precision, recall
    
    def generate_advanced_report(self, y_true, y_pred, report_filename, include_plots=True):
        """
        Generate a report with advanced metrics and optionally plots.
        
        Args:
            y_true (list): Ground truth annotations.
            y_pred (list): Predicted annotations.
            report_filename (str): Name of the report file.
            include_plots (bool): Whether to include plots in the report.
            
        Returns:
            dict: Dictionary of computed metrics.
        """
        # Ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Calculate advanced metrics
        metrics = self.calculate_advanced_metrics(y_true, y_pred)
        
        # Open the report file in write mode
        report_path = results_dir / report_filename
        try:
            with open(report_path, "w") as f:
                f.write("Advanced Object Detection Evaluation Metrics\n")
                f.write("==========================================\n\n")
                
                for metric_name, metric_value in metrics.items():
                    f.write(f"{metric_name}: {metric_value:.4f}\n")
                    
            logger.info(f"Advanced metrics report generated at {report_path}")
            
            # Generate plots if requested
            if include_plots:
                plots_dir = results_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                
                # Precision-recall curve
                pr_curve_path = plots_dir / "precision_recall_curve.png"
                self.plot_precision_recall_curve(y_true, y_pred, 0.5, str(pr_curve_path))
                
        except Exception as e:
            logger.error(f"Error generating advanced metrics report: {e}")
            
        return metrics
