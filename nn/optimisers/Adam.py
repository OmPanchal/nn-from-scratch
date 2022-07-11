import numpy as np
from nn.optimisers.Momentum import Momentum
from nn.optimisers.Optimiser import Optimiser
from nn.optimisers.RMSProp import RMSProp


class Adam(Optimiser):
	def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
		func = self.func
		self.beta1 = beta1
		self.beta2 = beta2

		self.epsilon = epsilon
		
		self.updateV = None
		self.updateM = None
		super().__init__(learning_rate, func)

	def func(self, grad, prev_grad):
		if type(prev_grad) == int: prev_grad = np.zeros(shape=(2))

		self.updateM, _ = (self.beta1 * prev_grad) + ((1 - self.beta1) * grad)
		self.updateV, _ = RMSProp().call(grad, prev_grad=prev_grad[1])

		grad_update_val = ((self.learning_rate * self.updateM) / np.sqrt(self.updateV + self.epsilon)) 

		return [self.updateM, self.updateV], grad_update_val 