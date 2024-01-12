# Model file which contains a model class in scikit-learn style
# Model class must have these 3 methods
# - __init__: initializes the model
# - fit: trains the model
# - predict: uses the model to perform predictions
#
# Created by: Ihsan Ullah
# Created on: 10 Jan, 2024

# ----------------------------------------
# Imports
# ----------------------------------------
from sklearn.linear_model import LogisticRegression


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
        self.clf = LogisticRegression()

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
        self.clf.fit(X, y)

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
        y = self.clf.predict(X)

        return y
