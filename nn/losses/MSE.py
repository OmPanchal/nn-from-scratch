import numpy as np
from nn.losses.Loss import Loss


class MSE(Loss):

    @staticmethod
    def MSE(pred, y): return (2 / len(y)) * np.sum(np.subtract(y, pred) ** 2)

    @staticmethod
    def MSE_prime(pred, y): return 2 * (pred - y) / np.size(y)

    def __init__(self):
        # loss = lambda pred, y: (2 / len(y)) * np.sum(np.subtract(y, pred) ** 2)
        # loss_prime = lambda pred, y: 2 * (pred - y) / np.size(y)
        super().__init__(MSE.MSE, MSE.MSE_prime)
