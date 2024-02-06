"""The Ingestion class is designed to handle the entire data ingestion process, 
    which includes loading, transforming, preparing, training a model, making predictions,
    and saving results
"""
# ------------------------------------------
# Imports
# ------------------------------------------
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime as dt
import ast
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch

import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()

import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# BASELINE 2 ###################################################
###############################################################


# ------------------------------------------
# Default Directories
# ------------------------------------------

""" Setting up the directory paths for input, output, and program files.
These directories will be used throughout the script.
"""
# ------------------------------------------
# Codabench Directories
# ------------------------------------------
# Root directory
root_dir = "/app"
# Input data directory to read training data from
input_dir = os.path.join(root_dir, "input_data")
# Input data directory to read test data from
reference_dir = os.path.join(root_dir, "reference_data")
# Output data directory to write predictions to
output_dir = os.path.join(root_dir, "output")
# Program directory
program_dir = os.path.join(root_dir, "program")
# Directory to read submitted submissions from
submission_dir = os.path.join(root_dir, "ingested_program")


sys.path.append(input_dir)
sys.path.append(output_dir)
sys.path.append(program_dir)
sys.path.append(submission_dir)

# ------------------------------------------
# Import Model
# ------------------------------------------
from model import Model



class Ingestion():
    """ Defining the Ingestion class responsible for handling data loading,
        transformation, preparation, model initialization, fitting, predicting,
        and result saving.
    """
    def __init__(self):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.model = None

    def start_timer(self):
        # Start the timer to measure script execution time
        self.start_time = dt.now()

    def stop_timer(self):
        #  Stop the timer when script execution is completed.
        self.end_time = dt.now()

    def get_duration(self):
        #  Calculate and return the total duration of script execution.
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def show_duration(self):
        # Display the total duration of script execution
        print("\n---------------------------------")
        print(f'[✔] Total duration: {self.get_duration()}')
        print("---------------------------------")

    def load_data(self):
        #  Load data from a CSV file into a Pandas DataFrame.
        print("[*] Loading Data")

        # data file path
        input_data_file = os.path.join(input_dir, 'input_data.csv')
        refer_data_file = os.path.join(reference_dir, 'reference_data.csv')

        # read data
        input_df = pd.read_csv(input_data_file)
        refer_df = pd.read_csv(refer_data_file)

        self.X_train = input_df.iloc[:, :-1].values
        self.y_train = input_df.iloc[:, -1].values

        self.X_test = refer_df.iloc[:, :-1].values
        self.y_test = refer_df.iloc[:, -1].values

    def get_train_data(self):
        # Return the training data for the model.
        return self.X_train, self.y_train

    def get_test_data(self):
        # Return the test data for evaluating the model.
        return self.X_test, self.y_test

    def init_submission(self):
        # Initialize the submitted model for training and prediction.
        print("[*] Initializing Submmited Model")
        self.model = Model()

    def fit_submission(self):
        # Train the submitted model on the training data.
        print("[*] Calling fit method of submitted model")
        X_train, y_train  = self.get_train_data()
        self.model.fit(X_train, y_train)

    def predict_submission(self):
        # Use the trained model to make predictions on the test data.
        print("[*] Calling predict method of submitted model")

        X_test, _ = self.get_test_data()
        self.y_test_hat = self.model.predict(X_test)

    def save_result(self):
        # Save the ingestion result
        print("[*] Saving ingestion result")

        _, y_test = self.get_test_data()
        ingestion_result_dict = {
            "predictions": self.y_test_hat.tolist(),
            "labels": y_test
        }
        result_file = os.path.join(output_dir, "result.json")
        with open(result_file, 'w') as f:
            f.write(json.dumps(ingestion_result_dict, indent=4))


if __name__ == '__main__':

    print("############################################")
    print("### Ingestion Program")
    print("############################################\n")

    # Init Ingestion
    ingestion = Ingestion()

    # Start timer
    ingestion.start_timer()

    # load train set
    ingestion.load_data()

    # transform data
    ingestion.transfrom_data()

    # prepare data
    ingestion.prepare_data()

    # initialize submission
    ingestion.init_submission()

    # fit submission
    ingestion.fit_submission()

    # predict submission
    ingestion.predict_submission()

    # save result
    ingestion.save_result()

    # Stop timer
    ingestion.stop_timer()

    # Show duration
    ingestion.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")
