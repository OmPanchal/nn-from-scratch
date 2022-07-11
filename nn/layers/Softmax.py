import numpy as np
from nn.layers.Layer import Layer


class Softmax(Layer):
	def __init__(self):
		self.output = None

	def forward(self, input: np.ndarray):	
		tmp = np.exp(input) + np.max(input)

		self.input = input
		self.output = tmp / np.sum(tmp)

		return self.output
		
	def backwards(self, error):
		n = np.size(self.output)
		tmp = np.tile(self.output, n)
		
		return np.dot(tmp * (np.identity(n) - tmp.T), error)

	def update(self, *args, **kwargs): ...