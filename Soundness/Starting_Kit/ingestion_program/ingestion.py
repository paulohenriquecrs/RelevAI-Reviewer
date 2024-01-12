# ------------------------------------------
# Imports
# ------------------------------------------
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime as dt
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer


import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()


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


sys.path.append(input_dir)
sys.path.append(output_dir)
sys.path.append(program_dir)
sys.path.append(submission_dir)

# ------------------------------------------
# Import Model
# ------------------------------------------
from model import Model


class Ingestion():

    def __init__(self):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.model = None

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def show_duration(self):
        print("\n---------------------------------")
        print(f'[✔] Total duration: {self.get_duration()}')
        print("---------------------------------")

    def load_data(self):
        """
        Loads data from csv file
        """
        print("[*] Loading Data")

        # files path
        train_data_file = os.path.join(input_dir, 'train.csv')
        test_data_file = os.path.join(input_dir, 'test.csv')
        train_labels_file = os.path.join(input_dir, 'train.labels')

        # read data
        self.train_df = pd.read_csv(train_data_file)
        self.test_df = pd.read_csv(test_data_file)

        # read labels
        with open(train_labels_file, 'r') as file:
            self.train_labels = [int(line.strip()) for line in file.readlines()]

    def transform_data(self):

        print("[*] Transforming Data")
        # pre-trained model
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

        # train context and references
        train_contexts = self.train_df['context_sentences'].tolist()
        train_references = (self.train_df['reference_title'] + " " + self.train_df['reference_abstract']).tolist()

        # test context and references
        test_contexts = self.test_df['context_sentences'].tolist()
        test_references = (self.test_df['reference_title'] + " " + self.test_df['reference_abstract']).tolist()

        # Compute train embeddings
        train_context_embeddings = self._compute_embeddings(train_contexts)
        train_reference_embeddings = self._compute_embeddings(train_references)

        # Compute test embeddings
        test_context_embeddings = self._compute_embeddings(test_contexts)
        test_reference_embeddings = self._compute_embeddings(test_references)

        # Calculate cosine similarity scores
        cosine_similarity = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        train_similarity_scores = [cosine_similarity(context, reference) for context, reference in zip(train_context_embeddings, train_reference_embeddings)]
        test_similarity_scores = [cosine_similarity(context, reference) for context, reference in zip(test_context_embeddings, test_reference_embeddings)]

        self.X_train = train_similarity_scores
        self.X_test = test_similarity_scores

    def get_train_data(self):
        return self.X_train, self.train_labels

    def get_test_data(self):
        return self.X_test

    def _compute_embeddings(self, texts):
        return self.embeddings_model.encode(texts, convert_to_tensor=False)

    def init_submission(self):
        print("[*] Initializing Submmited Model")
        self.model = Model()

    def fit_submission(self):
        print("[*] Calling fit method of submitted model")
        X_train, y_train = self.get_train_data()
        X_train = np.array(X_train).reshape(-1, 1)
        self.model.fit(X_train, y_train)

    def predict_submission(self):
        print("[*] Calling predict method of submitted model")
        X_test = self.get_test_data()
        X_test = np.array(X_test).reshape(-1, 1)
        self.y_test_hat = self.model.predict(X_test)

    def save_result(self):
        print("[*] Saving ingestion result")

        ingestion_result_dict = {
            "predictions": self.y_test_hat.tolist()
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
    ingestion.transform_data()

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
