import numpy as np
import torch


def init_scaler(scaler_type):
    if scaler_type == "standard":
        return StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")


class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X, mask=None):
        if mask is not None:
            X = np.ma.masked_array(X, ~mask.astype(bool))
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def to_cuda(self):
        self.mean = torch.from_numpy(self.mean).cuda()
        self.std = torch.from_numpy(self.std).cuda()
