import torch
from hyperfast import HyperFastClassifier
import numpy as np


class HyperFastModel:

    def __init__(self, **kwargs):
        """
        Initialize the model.
        kwargs will contain 'random_state', which we must 'pop' here,
        as it is not passed to the underlying HyperFastClassifier.
        """
        # HyperFast requires a 'device' argument
        # We automatically detect it here to keep the main script clean
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        kwargs.pop('random_state', None)

        # Add 'device' to kwargs
        kwargs['device'] = self.device

        # Instantiate the underlying model (kwargs now contains 'device')
        self.model = HyperFastClassifier(**kwargs)
        print(f"HyperFastModel initialized on device: {self.device}")

    def train_and_predict(self, X_train, y_train, X_test):
        """
        Train the model and return predictions for X_test.

        Returns:
        - y_pred: Class predictions (0 or 1)
        - y_pred_prob: Probability of class 1
        """
        # Ensure data are numpy arrays (the main script passes numpy arrays)
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.numpy()
            y_train = y_train.numpy()
            X_test = X_test.numpy()

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Get probabilities
        # Ensure y_pred_prob is a 1D array
        y_pred_prob = self.model.predict_proba(X_test)
        if y_pred_prob.ndim == 2:
            y_pred_prob = y_pred_prob[:, 1]

        return y_pred, y_pred_prob