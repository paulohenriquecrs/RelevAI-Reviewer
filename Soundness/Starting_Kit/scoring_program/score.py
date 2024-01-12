# ------------------------------------------
# Imports
# ------------------------------------------
import os
import numpy as np
import json
from datetime import datetime as dt
from sklearn.metrics import accuracy_score

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
        print(f"[✔] Total duration: {self.get_duration()}")
        print("---------------------------------")

    def load_ingestion_result(self):
        print("[*] Reading predictions")

        results_file = os.path.join(prediction_dir, "result.json")
        with open(results_file) as f:
            ingestion_result = json.load(f)
            self.y_test_hat = ingestion_result["predictions"]

        print("[✔]")

    def load_test_labels(self):
        print("[*] Reading test labels")

        test_labels_file = os.path.join(reference_dir, "test.labels")
        with open(test_labels_file, 'r') as file:
            self.y_test = [int(line.strip()) for line in file.readlines()]

        print("[✔]")

    def compute_scores(self):
        print("[*] Computing scores")

        accuracy = accuracy_score(self.y_test, self.y_test_hat)
        self._print(f"Accuracy Score: {accuracy}")

        self.scores_dict = {
            "accuracy": accuracy
        }

        print("[✔]")

    def write_scores(self):
        print("[*] Writing scores")

        with open(score_file, "w") as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))

        print("[✔]")

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

    # Load test labels
    scoring.load_test_labels()

    # Compute Scores
    scoring.compute_scores()

    # Write scores
    scoring.write_scores()

    # Stop timer
    scoring.stop_timer()

    # Show duration
    scoring.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Scoring Program executed successfully!")
    print("----------------------------------------------\n\n")
