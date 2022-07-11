import numpy as np
from nn.layers.Layer import Layer


class Dense(Layer):
    def __init__(self, input_units: int, output_units: int, activation=None):
        super(Dense, self).__init__(input_units, output_units)

        self.prev_grad_weights = 0
        self.prev_grad_biases = 0
        
        self.m = input_units

        self.weights = np.random.rand(output_units, input_units) - 0.5
        self.biases = np.random.rand(output_units, 1) - 0.5

        self.weights_grad = None
        self.error = None

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

    def backwards(self, error):
        if self.activation:
            activation_error = self.activation.backwards(error)
        else:
            activation_error = error
        self.error = activation_error

        self.weights_grad = np.dot(activation_error, self.input.T)

        return np.dot(self.weights.T, error)

    def update(self, learning_rate, update):
        grad_weights, update_weights = update(self.weights_grad, self.prev_grad_weights)
        self.prev_grad_weights = grad_weights

        grad_biases, update_biases = update(self.error, self.prev_grad_biases)
        self.prev_grad_biases = grad_biases

        self.weights -= update_weights
        self.biases -= update_biases
