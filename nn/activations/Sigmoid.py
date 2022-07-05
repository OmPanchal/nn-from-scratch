import numpy as np
from nn.activations.Activation import Activation


class Sigmoid(Activation):

	@staticmethod
	def sigmoid(x): return 1 / (1 + np.exp(-1 * x))

	@staticmethod
	def sigmoid_prime(x): return Sigmoid.sigmoid(x) * (1 - Sigmoid.sigmoid(x))

	def __init__(self):
		# activation = lambda x: 1 / (1 + np.exp(-1 * x))
		# activation_prime = lambda x: activation(x) * (1 - activation(x))
		super().__init__(Sigmoid.sigmoid, Sigmoid.sigmoid_prime)