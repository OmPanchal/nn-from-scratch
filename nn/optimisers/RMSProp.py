import numpy as np
from nn.optimisers.Optimiser import Optimiser


class RMSProp(Optimiser):
	def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
		func = self.func
		self.beta = beta
		self.epsilon = epsilon
		self.update = None
		super().__init__(learning_rate, func)

	def func(self, grad, prev_grad):
		self.update = (self.beta * prev_grad) + ((1 - self.beta) * (grad ** 2))
		return self.update, ((grad * self.learning_rate) / np.sqrt(self.update + self.epsilon))
