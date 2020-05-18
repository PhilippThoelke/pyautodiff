from pyautodiff import Variable
import random
import math

class MLP:
	def __init__(self, n_inputs, layer_sizes):
		self.weights = []
		self.bias = []
		for i in range(len(layer_sizes)):
			n_in = layer_sizes[i-1] if i > 0 else n_inputs
			n_neurons = layer_sizes[i]
			self.weights.append([[Variable(random.gauss(0, 1)) for _ in range(n_in)] for _ in range(n_neurons)])
			self.bias.append([Variable(0) for _ in range(n_neurons)])

	def __call__(self, x):
		for ws, bs in zip(self.weights, self.bias):
			x = [MLP._sigmoid(sum(xi * wi for xi, wi in zip(x, w)) + b) for w, b in zip(ws, bs)]
		return x

	def _sigmoid(x):
		return 1 / (1 + math.e ** -x)

	def variables(self):
		return [self.weights, self.bias]

	def apply_gradient(self, gradient, lr=1):
		for layer_weights, layer_gradient in zip(self.weights, gradient[0]):
			for neuron_weights, neuron_gradient in zip(layer_weights, layer_gradient):
				for w, grad in zip(neuron_weights, neuron_gradient):
					w.set(float(w) - lr * grad)
		for layer_bias, layer_gradient in zip(self.bias, gradient[1]):
			for neuron_bias, neuron_gradient in zip(layer_bias, layer_gradient):
				neuron_bias.set(float(neuron_bias) - lr * neuron_gradient)

if __name__ == '__main__':
	mlp = MLP(2, (2, 1))
	# example data: logical XOR
	data = [[0, 0], [1, 0], [0, 1], [1, 1]]
	labels = [0, 1, 1, 0]

	n_epochs = 500
	print('Training...')
	for epoch in range(n_epochs):
		epoch_loss = 0
		for sample, lbl in zip(data, labels):
			pred = mlp(sample)
			loss = (pred[0] - lbl) ** 2
			epoch_loss += float(loss)
			mlp.apply_gradient(loss.gradient(mlp.variables()))

		if epoch % (n_epochs // 5) == 0:
			print(f'Loss is {epoch_loss / len(data):.3f} in epoch {epoch + 1}')

	print('\nTesting...')
	for sample in data:
		pred = float(mlp(sample)[0])
		print(f'Prediction for {sample}: {pred:.3f} --> {pred > 0.5}')
