import numpy as np


class Activation:
    def __init__(self, activation, activation_prime):
        self.input = None
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backwards(self, error, learning_rate):
        return np.multiply(error, self.activation_prime(self.input))