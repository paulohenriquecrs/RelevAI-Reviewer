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
from transformers import BertForSequenceClassification
import torch
from tqdm import tqdm

from tqdm.notebook import tqdm
# from sklearn.naive_bayes import MultinomialNB


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
        self.id2label = {0: "least_relevant", 1: "second_least_relevant", 2: "second_most_relevant", 3: "most_relevant"}
        self.label2id = {"least_relevant": 0, "second_least_relevant": 1, "second_most_relevant": 2, "most_relevant": 3}
        
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            problem_type="multi_label_classification",
            num_labels=4,
            id2label=self.id2label,
            label2id=self.label2id
            )

        # set device (cpu for local machines)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

    def fit(self, dataloader, num_epochs=1):
        """
        This function trains the model provided training data

        Parameters
        ----------
        train_dataloader:   Pytorch Dataloader with training data
        num_epochs:         Integer representing the number of epochs fro training

        Returns
        -------
        None
        """

        print("[*] - Training Classifier on the train set")
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
    
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
    
                self.optimizer.zero_grad()
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
    
                loss.backward()
                self.optimizer.step()
    
            avg_loss = total_loss / len(dataloader)
            print(f"Average training loss: {avg_loss}")


    def predict(self, dataloader):

        """
        This function predicts labels on test data.

        Parameters
        ----------
        X: Pytorch Dataloader with test data

        Returns
        -------
        y_pred: List of predicted labels
        y_true: List of true test labels
        """

        print("[*] - Predicting test set using trained Classifier")
        self.model.eval()
        predictions = []
        true_labels = []
        
        for batch in dataloader:
            inputs, labels = batch
            with torch.no_grad():
                outputs = self.model(inputs)
        
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(labels.tolist())

        return predictions, true_labels
