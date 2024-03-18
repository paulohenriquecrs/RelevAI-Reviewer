""" The `Model` file encapsulates a model class structured in the style of scikit-learn. 
    This model class is equipped with three essential methods:
    - __init__: Initializes the model.
    - fit: Trains the model.
    - predict: Utilizes the model to make predictions.
Created by: Ihsan Ullah
Created on: 10 Jan, 2024
Updated by: Paulo Couto
Updated on: 18 Mar, 2024
"""


# ----------------------------------------
# Imports
# ----------------------------------------
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime as dt
import ast
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer


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
        regressor = SVC(kernel='rbf', gamma = 'scale', C = 1)
        self.pipeline = Pipeline([('pca', pca),
                                  ('regressor', regressor)
                                 ])


    def _get_embeddings(self, text1, text2):
        """
        Generates embeddings for two texts.

        :param text1: First text string.
        :param text2: Second text string.
        :return: Tuple of embeddings for text1 and text2.
        """
        embedding1 = self.embeddings_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.embeddings_model.encode(text2, convert_to_tensor=True)
        return embedding1.cpu(), embedding2.cpu()

    def prepare_data(self, df, str = "Training/Testing"):

        print("[*] Prepare Data for " + str)
        
        model_name = 'paraphrase-MiniLM-L6-v2'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings_model = SentenceTransformer(model_name, device=device)
    
        # Create embeddings
        
        df['embeddings'] = df.progress_apply(lambda row: self._get_embeddings(row['prompt'], row['text']), axis=1)
    
        X = df['embeddings'].tolist()
    
        # Convert embeddings from tuples to concatenated arrays
        X = [torch.abs(embeddings[0] - embeddings[1]).numpy() for embeddings in X]

        return X

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
