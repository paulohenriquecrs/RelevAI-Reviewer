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
from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch

from torch.utils.data import DataLoader, TensorDataset

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
        print(f'Total duration: {self.get_duration()}')
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

    
    def pad_sequence_to_max_length(self, sequence, padding_value=0):
        """
        this functions pads an input sequence
        """
        padded_sequence = sequence + [padding_value] * (self.max_length - len(sequence))
        return padded_sequence


    def prepare_data(self):
    
        print("[*] Prepare Data for Training")
    
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
        least_relevant_df = pd.DataFrame()
        least_relevant_df['input'] = self.df.progress_apply(lambda row: tokenizer.encode(row['prompt'] + " " + row['least_relevant_text'], truncation=True), axis=1)
        least_relevant_df['label'] = 0
        
        second_least_relevant_df = pd.DataFrame()
        second_least_relevant_df['input'] = self.df.progress_apply(lambda row: tokenizer.encode(row['prompt'] + " " + row['second_least_relevant_text'], truncation=True), axis=1)
        second_least_relevant_df['label'] = 1
        
        second_most_relevant_df = pd.DataFrame()
        second_most_relevant_df['input'] = self.df.progress_apply(lambda row: tokenizer.encode(row['prompt'] + " " + row['second_most_relevant_text'], truncation=True), axis=1)
        second_most_relevant_df['label'] = 2
        
        most_relevant_df = pd.DataFrame()
        most_relevant_df['input'] = self.df.progress_apply(lambda row: tokenizer.encode(row['prompt'] + " " + row['most_relevant_text'], truncation=True), axis=1)
        most_relevant_df['label'] = 3
    
        # combine all 4 dataframes
        tokenized_df = pd.concat([least_relevant_df, second_least_relevant_df, second_most_relevant_df, most_relevant_df], ignore_index=True)
        
        tokenized_df['label_onehot'] = tokenized_df['label'].apply(lambda x: list(np.eye(4)[x]))
    
        # shuffle combined dataframe
        tokenized_df = tokenized_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
        # pad the sequences
        self.max_length=tokenized_df['input'].apply(len).max()
        tokenized_df['input_padded'] = tokenized_df['input'].apply(lambda x: self.pad_sequence_to_max_length(x))
        
        X = torch.tensor(tokenized_df['input_padded'].tolist())
        y = torch.tensor(tokenized_df['label_onehot'].tolist())
    
        dataset = TensorDataset(X, y)
        self.train_dataset, self.test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    def get_train_data(self):
        return self.train_dataset
  
    def get_test_data(self):
        return self.test_dataset
  
    def show_random_sample(self):
        random_sample_index = np.random.randint(0, len(self.df))
    
        print("Prompt:\n", self.df.iloc[random_sample_index]['prompt'], "...\n")
        print("Most Relevant Text:\n", self.df.iloc[random_sample_index]['most_relevant_text'][:300], "...\n")
        print("Second Most Relevant Text:\n", self.df.iloc[random_sample_index]['second_most_relevant_text'][:300], "...\n")
        print("Second Least Relevant Text:\n", self.df.iloc[random_sample_index]['second_least_relevant_text'][:300], "...\n")
        print("Least Relevant Text:\n", self.df.iloc[random_sample_index]['least_relevant_text'][:300], "...\n")
    def init_submission(self):
        print("[*] Initializing Submmited Model")
        self.model = Model()

    def fit_submission(self):
        print("[*] Calling fit method of submitted model")
        train_dataloader = DataLoader(self.get_train_data(), batch_size=8, shuffle=True)
        num_epochs=1
        self.model.fit(train_dataloader, num_epochs)

    def predict_submission(self):
        print("[*] Calling predict method of submitted model")

        test_dataloader = DataLoader(self.get_test_data(), batch_size=8, shuffle=False)
        self.y_test_hat, self.y_test_truth = self.model.predict(test_dataloader)

    def save_result(self):
        print("[*] Saving ingestion result")

        y_test_ = np.array(self.y_test_truth)
        y_test = np.argmax(y_test_, axis=1)
        ingestion_result_dict = {
            "predictions": self.y_test_hat.tolist(),
            "labels": y_test.tolist()
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
    print("Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")
