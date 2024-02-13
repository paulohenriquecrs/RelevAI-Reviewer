# ------------------------------------------
# Imports
# ------------------------------------------
import os
import numpy as np
import json
from datetime import datetime as dt
from sklearn.metrics import classification_report
from scipy.stats import kendalltau

import random
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ------------------------------------------
# Default Directories
# ------------------------------------------
# root directory
module_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(module_dir)
# Directory to output computed score into
output_dir = os.path.join(root_dir, "scoring_output")
# reference data (test labels)
reference_dir = os.path.join(root_dir, "sample_data")
# submitted/predicted lables
prediction_dir = os.path.join(root_dir, "sample_result_submission")
# score file to write score into
score_file = os.path.join(output_dir, "scores.json")
# html file to write score and figures into
html_file = os.path.join(output_dir, 'detailed_results.html')


class Scoring:
    def __init__(self):
        # Initialize class variables
        self.start_time = None
        self.end_time = None

        self.scores_dict = {}

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
        print(f"[OK] Total duration: {self.get_duration()}")
        print("---------------------------------")

    def load_ingestion_result(self):
        print("[*] Reading predictions")

        results_file = os.path.join(prediction_dir, "result.json")
        with open(results_file) as f:
            ingestion_result = json.load(f)
            self.y_test_hat = ingestion_result["predictions"]
            self.y_test = ingestion_result["labels"]

        print("[OK]")

    def compute_scores(self):
        print("[*] Computing scores")

        # Classification report
        self._print(classification_report(self.y_test, self.y_test_hat))

        k_tau, _ = kendalltau(self.y_test, self.y_test_hat)
        self._print(f"Kendall's Tau: {k_tau}")

        self.scores_dict = {
            "k_tau": k_tau
        }

        print("[OK]")

    def write_scores(self):
        print("[*] Writing scores")

        with open(score_file, "w") as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))

        print("[OK]")

    def write_html(self, content):
        with open(html_file, 'a', encoding="utf-8") as f:
            f.write(content)

    def _print(self, content):
        print(content)
        self.write_html(content + "<br>")


if __name__ == "__main__":
    print("############################################")
    print("### Scoring Program")
    print("############################################\n")

    # Init scoring
    scoring = Scoring()

    # Start timer
    scoring.start_timer()

    # Load ingestions results
    scoring.load_ingestion_result()

    # Compute Scores
    scoring.compute_scores()

    # Write scores
    scoring.write_scores()

    # Stop timer
    scoring.stop_timer()

    # Show duration
    scoring.show_duration()

    print("\n----------------------------------------------")
    print("[OK] Scoring Program executed successfully!")
    print("----------------------------------------------\n\n")
