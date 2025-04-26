# ********************* Code for various metrics for sentiment analysis ***********************
# This code will generate accuracy score, classification report, confusion matrix, precision, recall, and F1 score.

# Import necessary libraries
import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Metrics:
    """
    A class for calculating various performance metrics for sentiment analysis models.
    
    This class provides methods to calculate accuracy, precision, recall, F1 score,
    and other metrics for evaluating model performance.
    """

    def __init__(self):
        """Initialize the Metrics class."""
        pass

    def accuracy(self, y_true, y_pred):
        """
        Calculate the accuracy of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - accuracy (float): Accuracy of the model.
        """
        return accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred, average='binary'):
        """
        Calculate the precision of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.
        - average (str): The averaging method. Defaults to 'binary'.

        Returns:
        - precision (float): Precision of the model.
        """
        return precision_score(y_true, y_pred, average=average)

    def recall(self, y_true, y_pred, average='binary'):
        """
        Calculate the recall of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.
        - average (str): The averaging method. Defaults to 'binary'.

        Returns:
        - recall (float): Recall of the model.
        """
        return recall_score(y_true, y_pred, average=average)

    def f1(self, y_true, y_pred, average='binary'):
        """
        Calculate the F1 score of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.
        - average (str): The averaging method. Defaults to 'binary'.

        Returns:
        - f1 (float): F1 score of the model.
        """
        return f1_score(y_true, y_pred, average=average)
        
    def confusion_matrix(self, y_true, y_pred):
        """
        Calculate the confusion matrix for the model.
        
        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.
        
        Returns:
        - conf_matrix (array): Confusion matrix.
        """
        return confusion_matrix(y_true, y_pred)
    
    def classification_report(self, y_true, y_pred, target_names=None):
        """
        Generate a classification report including precision, recall, F1-score.
        
        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.
        - target_names (list, optional): List of target names.
        
        Returns:
        - report (str): Text summary of the precision, recall, F1 score.
        """
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def evaluate(self, y_true, y_pred, target_names=None):
        """
        Evaluate the model and return all metrics as a dictionary.
        
        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.
        - target_names (list, optional): List of target names.
        
        Returns:
        - metrics (dict): Dictionary containing all metrics.
        """
        try:
            logger.info("Calculating evaluation metrics")
            
            # Convert inputs to numpy arrays for consistent handling
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Determine if binary or multiclass
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            is_binary = len(unique_classes) <= 2
            average_method = 'binary' if is_binary else 'weighted'
            
            metrics = {
                'accuracy': self.accuracy(y_true, y_pred),
                'precision': self.precision(y_true, y_pred, average=average_method),
                'recall': self.recall(y_true, y_pred, average=average_method),
                'f1': self.f1(y_true, y_pred, average=average_method),
                'confusion_matrix': self.confusion_matrix(y_true, y_pred).tolist(),
                'classification_report': self.classification_report(y_true, y_pred, target_names=target_names)
            }
            
            logger.info(f"Evaluation metrics calculated: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Test with sample data
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1]
    
    metrics = Metrics()
    result = metrics.evaluate(y_true, y_pred, target_names=['Negative', 'Positive'])
    
    print(f"Accuracy: {result['accuracy']}")
    print(f"Precision: {result['precision']}")
    print(f"Recall: {result['recall']}")
    print(f"F1 Score: {result['f1']}")
    print("Confusion Matrix:")
    print(np.array(result['confusion_matrix']))
    print("\nClassification Report:")
    print(result['classification_report'])


