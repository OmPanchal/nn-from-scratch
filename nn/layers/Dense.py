import numpy as np
from nn.layers.Layer import Layer


class Dense(Layer):
    def __init__(self, input_units: int, output_units: int, activation=None):
        super(Dense, self).__init__(input_units, output_units)

        self.m = input_units

        self.weights = np.random.rand(output_units, input_units) - 0.5
        self.biases = np.random.rand(output_units, 1) - 0.5

        if activation is not None:
            self.activation = activation
        else:
            self.activation = None

    def forward(self, input: np.ndarray):
        self.input = input
        output = np.dot(self.weights, input) + self.biases

        if self.activation:
            activation_output = self.activation.forward(output)
            return activation_output
        else:
            return output

    def backwards(self, error, learning_rate):
        if self.activation:
            activation_error = self.activation.backwards(error, learning_rate)
        else:
            activation_error = error

        weights_grad = np.dot(activation_error, self.input.T)

        self.weights -= learning_rate * weights_grad
        self.biases -= learning_rate * error

        return np.dot(self.weights.T, error)

