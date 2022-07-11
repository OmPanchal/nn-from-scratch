from nn.optimisers.Optimiser import Optimiser


class Momentum(Optimiser):
	def __init__(self, learning_rate=0.01, beta=0.9):
		func = self.func
		self.beta = beta
		self.update = None
		super().__init__(learning_rate, func)

	def func(self, grad, prev_grad):
		self.update = (self.beta * prev_grad) + (self.learning_rate * grad)
		return self.update, self.update