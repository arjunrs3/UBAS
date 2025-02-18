"""
quant_regressor.py
==================
A simple regressor class which defines a neural net quantile regressor using skorch
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, ProgressBar
from sklearn.base import BaseEstimator, RegressorMixin


class QuantileLoss(nn.Module):
    """Tilted loss function for quantile regression"""
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)

    def forward(self, y_pred, y_true):
        error = y_true.view(-1, 1) - y_pred
        return torch.mean(torch.max(self.quantiles * error, (self.quantiles - 1) * error))


class MLP(nn.Module):
    """Constructor class for a basic MLP"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(num_layers-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


class QuantNN(BaseEstimator, RegressorMixin):
    """A Neural Network which can perform quantile regression"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=3, quantiles=np.array([0.05, 0.95]), lr=0.001,
                 max_epochs=500, batch_size=32,
                 patience=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.quantiles = quantiles
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience

        self.net = NeuralNetRegressor(
            module=MLP,
            module__input_dim=self.input_dim,
            module__hidden_dim=self.hidden_dim,
            module__output_dim=self.output_dim,
            module__num_layers=self.num_layers,
            criterion=QuantileLoss,
            criterion__quantiles=self.quantiles,
            optimizer=optim.Adam,
            optimizer__lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            train_split=None,
            callbacks=[ProgressBar()]
        )

    def fit(self, X, y):
        """Fit method to conform to scikit-api"""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        self.net.fit(X, y)
        return self

    def predict(self, X):
        """Predict method to conform to scikit-api"""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        quantiles = self.net.predict(X).T
        return self.net.predict(X).T
