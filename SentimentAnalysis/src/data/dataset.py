# Code for preparing dataset for training, testing, and validation

# Import necessary libraries
import pandas as pd
import numpy as np
from preprocess import normalization
from gensim.models import Word2Vec


# class for dataset preparation

class Sentiment_Analysis_Dataset:

    def __init__(self, csv_file_path):
        # Load data from the provided CSV file
        self.data = pd.read_csv(csv_file_path)
        # Assuming the text column is named 'text'
        self.reviews = self.data[['text']]

    def split_dataset(self, train_size=0.8, test_size=0.1, val_size=0.1):
        # Split the dataset into training, testing, and validation sets
        train, test, val = np.split(self.reviews.sample(frac=1), [int(train_size * len(self.reviews)),
                                                                  int((train_size + test_size) * len(self.reviews))])
        return train, test, val

    def k_fold_split(self, k=5):
        # Split the dataset into k-folds
        folds = []
        for i in range(k):
            fold = self.reviews.sample(frac=1 / k)
            folds.append(fold)
        return folds
