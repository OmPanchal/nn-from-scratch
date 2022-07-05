from cmath import tanh
from nn.activations.Activation import Activation
import numpy as np


class Tanh(Activation):

    @staticmethod
    def tanh(x): return np.tanh(x)

    @staticmethod
    def tanh_prime(x): return 1 - np.tanh(x) ** 2

    def __init__(self):
        # activtion = lambda x: np.tanh(x)
        # activation_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(Tanh.tanh, Tanh.tanh_prime)
