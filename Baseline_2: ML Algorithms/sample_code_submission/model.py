""" The `Model` file encapsulates a model class structured in the style of scikit-learn. 
    This model class is equipped with three essential methods:
    - __init__: Initializes the model.
    - fit: Trains the model.
    - predict: Utilizes the model to make predictions.
Created by: Ihsan Ullah
Created on: 10 Jan, 2024
"""


# ----------------------------------------
# Imports
# ----------------------------------------
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import numpy as np
import random
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ----------------------------------------
# Model Class
# ----------------------------------------
class Model:

    def __init__(self):
        """
        This is a constructor for initializing classifier

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("[*] - Initializing Classifier")
        pca = PCA(n_components = None)
        regressor = SVR(kernel='rbf', gamma = 'scale', C = 1)
        self.pipeline = Pipeline([('pca', pca),
                                  ('regressor', regressor)
                                 ])

    def fit(self, X, y):
        """
        This function trains the model provided training data

        Parameters
        ----------
        X: 2D numpy array
            training data matrix of dimension num_train_examples * num_features
            each column is a feature and each row a datapoint
        y: 1D numpy array
            training label matrix of dimension num_train_samples

        Returns
        -------
        None
        """

        print("[*] - Training Classifier on the train set")
        self.pipeline.fit(X, y)

    def predict(self, X):

        """
        This function predicts labels on test data.

        Parameters
        ----------
        X: 2D numpy array
            test data matrix of dimension num_test_examples * num_features
            each column is a feature and each row a datapoint

        Returns
        -------
        y: 1D numpy array
            predicted labels
        """

        print("[*] - Predicting test set using trained Classifier")
        y = self.pipeline.predict(X)

        return y
