from torch import Tensor as T

EPSILON = 1e-9


def masked_mae(y_true: T, y_pred: T, mask: T) -> T:
    maes = (mask * (y_true - y_pred)).abs().sum()
    if mask.sum() == 0:
        return mask.sum()
    return maes / (mask.sum() + EPSILON)


def masked_mse(y_true: T, y_pred: T, mask: T) -> T:
    mse = (mask * (y_true - y_pred) ** 2).sum()
    return mse / (mask.sum() + EPSILON)
