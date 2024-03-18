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
# Default Directories
# ------------------------------------------
# Root directory
module_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(module_dir)
# Input data directory to read training and test data from
input_dir = os.path.join(root_dir, "sample_data")
# Output data directory to write predictions to
output_dir = os.path.join(root_dir, "sample_result_submission")
# Program directory
program_dir = os.path.join(root_dir, "ingestion_program")
# Directory to read submitted submissions from
submission_dir = os.path.join(root_dir, "sample_code_submission")


"""
# ------------------------------------------
# Codabench Directories
# ------------------------------------------
# Root directory
root_dir = "/app"
# Input data directory to read training data from
input_dir = os.path.join(root_dir, "input_data")
# Output data directory to write predictions to
output_dir = os.path.join(root_dir, "output")
# Program directory
program_dir = os.path.join(root_dir, "program")
# Directory to read submitted submissions from
submission_dir = os.path.join(root_dir, "ingested_program")
"""

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
        print(f'[OK] Total duration: {self.get_duration()}')
        print("---------------------------------")

    def load_data(self):
        """
          Loads data from csv file
        """
        print("[*] Loading Data")
    
        # data file path
        data_file = os.path.join(input_dir, 'relevance_sample_data.csv')
        
        # read data
        self.df = pd.read_csv(data_file)


    def _text_to_dict(self, text):
        """
        Converts a text string into a dictionary.
    
        :param text: A string representation of a dictionary.
        :return: A dictionary object if conversion is successful, otherwise {}.
        """
        try:
          return ast.literal_eval(text)
        except:
          return {}  # Return an empty dictionary in case of an error
    
    def _dict_to_paragraphs(self, dictionary):
        """
        Converts a dictionary into a string of paragraphs.
    
        :param dictionary: A dictionary.
        :return: A string composed of paragraphs based on the dictionary's key-value pairs.
        """
        text = ''
        for i, (k, v) in enumerate(dictionary.items()):
            text += k.capitalize() + '\n' + v + '\n'
        return text
  
    def transfrom_data(self):
    
        print("[*] Transforming Data")
        
        # Convert to dictionary
        self.df['most_relevant_dict'] = self.df['most_relevant'].apply(self._text_to_dict)
        self.df['second_most_relevant_dict'] = self.df['second_most_relevant'].apply(self._text_to_dict)
        self.df['second_least_relevant_dict'] = self.df['second_least_relevant'].apply(self._text_to_dict)
        self.df['least_relevant_dict'] = self.df['least_relevant'].apply(self._text_to_dict)
    
    
        # Convert from dictionary to text
        self.df['most_relevant_text'] = self.df['most_relevant_dict'].apply(self._dict_to_paragraphs)
        self.df['second_most_relevant_text'] = self.df['second_most_relevant_dict'].apply(self._dict_to_paragraphs)
        self.df['second_least_relevant_text'] = self.df['second_least_relevant_dict'].apply(self._dict_to_paragraphs)
        self.df['least_relevant_text'] = self.df['least_relevant_dict'].apply(self._dict_to_paragraphs)
      
    def prepare_data(self):
    
        print("[*] Prepare Data for Training")
        
    
        # Label the Data
        self.df['most_relevant_label'] = 3
        self.df['second_most_relevant_label'] = 2
        self.df['second_least_relevant_label'] = 1
        self.df['least_relevant_label'] = 0
    
        X = pd.DataFrame()
        X["prompt"] = self.df['prompt'].tolist() + self.df['prompt'].tolist() + self.df['prompt'].tolist() + self.df['prompt'].tolist()
        X["text"] = self.df['most_relevant_text'].tolist() + self.df['second_most_relevant_text'].tolist() + self.df['second_least_relevant_text'].tolist() + self.df['least_relevant_text'].tolist()
        y = self.df['most_relevant_label'].tolist() + self.df['second_most_relevant_label'].tolist() + self.df['second_least_relevant_label'].tolist() + self.df['least_relevant_label'].tolist()
    
        # Shuffle X and y
        X, y = shuffle(X, y, random_state=42)
    
        # train test split
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    def get_train_data(self):
        return self.X_train, self.y_train
  
    def get_test_data(self):
        return self.X_test, self.y_test

    def init_submission(self):
        # Initialize the submitted model for training and prediction.
        print("[*] Initializing Submmited Model")
        self.model = Model()
        self.X_train = self.model.prepare_data(self.X_train_raw, "Training")
        self.X_test = self.model.prepare_data(self.X_test_raw, "Testing")
        
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
    print("[OK] Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")
