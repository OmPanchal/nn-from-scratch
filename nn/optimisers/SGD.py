import numpy as np
from nn.optimisers.Optimiser import Optimiser


class SGD(Optimiser):

	# @staticmethod

	def __init__(self, learning_rate=0.1):
		func = self.func
		self.update = None
		super().__init__(learning_rate, func)

	def func(self, grad, *args):
		self.update = self.learning_rate * grad 
		return ..., self.update