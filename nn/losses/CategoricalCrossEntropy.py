import numpy as np
from nn.losses.Loss import Loss


class CategoricalCrossEntropy(Loss):

	@staticmethod
	def categorical_cross_entropy(pred, y): return -1 * (np.sum(y * np.log(pred + 1e-8)))
	
	@staticmethod
	def categorical_cross_entropy_prime(pred, y): return pred - y

	def __init__(self):
		# loss = lambda pred, y: -1 * (np.sum(y * np.log(pred + 1e-8)))
		# loss_prime = lambda pred, y: pred - y
		super().__init__(CategoricalCrossEntropy.categorical_cross_entropy, CategoricalCrossEntropy.categorical_cross_entropy_prime)
