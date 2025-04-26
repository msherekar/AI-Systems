import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
import datetime
import os
import pickle

class Metrics:
    def __init__(self, model_name='', model_dir=None):
        self.model_name = model_name
        self.report_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.report_path = f"{model_name}_report_{self.report_date}.txt"
        #self.model = None
        #self.models = {}
        
        if model_dir is None:
            self.model_dir = os.path.dirname(__file__)  # Directory of the current Python file
        else:
            self.model_dir = model_dir


    def generate_report(self, y_prediction, y_label):
        """
        Generate report and save to specified directory using sklearn metrics.
        """
        precision = precision_score(y_label, y_prediction)
        recall = recall_score(y_label, y_prediction)
        f1 = f1_score(y_label, y_prediction)

        # Write report to file
        with open(self.report_path, 'w') as f:
            f.write(f"Model: {self.model_name}\n")
            f.write("Model Metrics:\n")
            f.write(f"Precision: {precision:.2f}\n")
            f.write(f"Recall: {recall:.2f}\n")
            f.write(f"F1 Score: {f1:.2f}\n")

        print(f"Report generated and saved to {self.report_path}")
        
    def load_model(self, model_type):
        """
        Load a trained model of a specific type.

        Parameters:
        - model_type (str): Type of model to load.

        Returns:
        - loaded_model: Loaded model object.
        """
        # Construct full path to the model file
        full_model_path = os.path.join(self.model_dir, f"{model_type}_best_model.pkl")

        # Check if the model file exists
        if not os.path.exists(full_model_path):
            raise ValueError(f"No model of type {model_type} found.")

        # Load the model from the specified path
        with open(full_model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        return loaded_model
    
 

        
    

        
    
     
     
