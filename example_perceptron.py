from pyautodiff import Variable
import random
import math

class Neuron:
	def __init__(self, n_in):
		self.bias = Variable(0)
		self.weights = [Variable(random.gauss(0, 1)) for _ in range(n_in)]

	def __call__(self, x):
		return Neuron._sigmoid(sum(xi * w for xi, w in zip(x, self.weights)) + self.bias)

	def _sigmoid(x):
		return 1 / (1 + math.e ** -x)

	def variables(self):
		return self.weights + [self.bias]

	def apply_gradient(self, gradient, lr=1):
		for var, grad in zip(self.variables(), gradient):
			var.set(float(var) - lr * grad)

if __name__ == '__main__':
	perceptron = Neuron(2)
	# example data: logical AND
	data = [[0, 0], [1, 0], [0, 1], [1, 1]]
	labels = [0, 0, 0, 1]

	n_epochs = 250
	print('Training...')
	for epoch in range(n_epochs):
		epoch_loss = 0
		for sample, lbl in zip(data, labels):
			pred = perceptron(sample)
			loss = (pred - lbl) ** 2
			epoch_loss += float(loss)
			perceptron.apply_gradient(loss.gradient(perceptron.variables()))

		if epoch % (n_epochs // 5) == 0:
			print(f'Loss is {epoch_loss / len(data):.3f} in epoch {epoch + 1}')

	print('\nTesting...')
	for sample in data:
		pred = float(perceptron(sample))
		print(f'Prediction for {sample}: {pred:.3f} --> {pred > 0.5}')
