from nn.activations.Activation import Activation
import numpy as np


class Relu(Activation):

    @staticmethod
    def relu(x): return np.maximum(0, x)

    @staticmethod
    def relu_prime(x): return x > 1

    def __init__(self):
        # activation = lambda x: np.maximum(0, x)
        # activation_prime = lambda x: x > 1
        super().__init__(Relu.relu, Relu.relu_prime)