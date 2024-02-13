from transformers import BertForSequenceClassification
import torch
from tqdm import tqdm

from tqdm.notebook import tqdm

# from sklearn.naive_bayes import MultinomialNB
import random
import torch
import numpy as np

import torch.nn as nn

from scipy.stats import rankdata


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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            problem_type="multi_label_classification",
            num_labels=3,
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

    def fit(self, dataloader, num_epochs=1):
        """
        This function trains the model provided training data

        Parameters
        ----------
        train_dataloader:   Pytorch Dataloader with training data
        num_epochs:         Integer representing the number of epochs for training

        Returns
        -------
        None
        """

        print("[*] - Training Classifier on the train set")
        loss_store = []
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                logits = outputs.logits
                loss = nn.BCEWithLogitsLoss()(
                    logits.view(-1), labels.view(-1)
                )  # combine BCE and sigmoid

                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / len(dataloader)
            print(f"Average training loss: {avg_loss}")
            loss_store.append(avg_loss)

        return loss_store

    def _group_elements(self, dataloader):
        groups = []
        for inputs, labels in dataloader:
            group = [(inp, label) for inp, label in zip(inputs, labels)]
            groups.append(group)
        return groups

    def _bootstrap_groups(self, groups, num_samples=None):
        while True:
            # If num_samples is not provided, use the length of the groups
            if num_samples is None:
                num_samples = len(groups)

            # Sample indices with replacement
            sampled_indices = [
                random.randint(0, len(groups) - 1) for _ in range(num_samples)
            ]

            # Yield the sampled groups
            yield [groups[i] for i in sampled_indices]

    def predict(self, dataloader):
        """
        This function predicts labels on test data.

        Parameters
        ----------
        dataloader: Pytorch Dataloader with test data

        Returns
        -------
        predictions: List of predicted labels
        true_labels: List of true test labels
        """

        print("[*] - Predicting test set using trained Classifier")
        self.model.eval()
        predictions = []
        true_labels = []
        # Group the elements of the list
        groups = list(self._group_elements(dataloader))

        # Bootstrap the groups
        bootstrapped_groups = self._bootstrap_groups(groups)
        test_data = next(bootstrapped_groups)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for batch in test_data:
            inputs = torch.stack([inp for inp, _ in batch])
            labels = torch.stack([label for _, label in batch])
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = self.model(inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            s = torch.sum(probs, dim=1).tolist()
            ranked = rankdata(s) - 1
            predictions.extend(ranked)
            true_labels.extend(torch.sum(labels, dim=1).tolist())

        return predictions, true_labels
