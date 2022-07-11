import numpy as np


class Layer(object):
    def __init__(self, input_units, output_units):
        self.input = None
        self.input_units = input_units
        self.output_units = output_units

    def forward(self, input: np.ndarray): ...

    def backwards(self, error, learning_rate, optimiser): ...

