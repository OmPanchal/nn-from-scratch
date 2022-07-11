import numpy as np


class Optimiser(object):
	def __init__(self, learning_rate, func):
		self.learning_rate = learning_rate
		self.func = func

	def call(self, grad, prev_grad):
		return self.func(grad, prev_grad)