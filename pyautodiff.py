from math import log

class Operation:
	def __init__(self, v1, v2=0.0, op='VAR'):
		self.v1 = v1
		self.v2 = v2
		self.op = op

	def gradient(self, vars):
		if hasattr(vars, '__iter__'):
			return [self.gradient(v) for v in vars]

		g1 = self.v1.gradient(vars) if isinstance(self.v1, Operation) else 0.0
		g2 = self.v2.gradient(vars) if isinstance(self.v2, Operation) else 0.0

		if self.op == 'ADD':
			return float(g1 + g2)
		elif self.op == 'SUB':
			return float(g1 - g2)
		elif self.op == 'MUL':
			return float(g1 * self.v2 + self.v1 * g2)
		elif self.op == 'DIV':
			return float((g1 * self.v2 - self.v1 * g2) / self.v2 ** 2)
		elif self.op == 'POW':
			return (float(self.v2) * float(self.v1) ** (float(self.v2) - 1) * g1) + \
				   (log(float(self.v1)) * float(self.v1) ** float(self.v2) * g2)
		elif self.op == 'VAR':
			return 1.0 if self == vars else 0.0
		else:
			return NotImplementedError(f'Operation {self.op} undefined')

	def __add__(self, other):
		return Operation(self, other, 'ADD')

	def __radd__(self, other):
		return Operation(other, self, 'ADD')

	def __sub__(self, other):
		return Operation(self, other, 'SUB')

	def __rsub__(self, other):
		return Operation(other, self, 'SUB')

	def __mul__(self, other):
		return Operation(self, other, 'MUL')

	def __rmul__(self, other):
		return Operation(other, self, 'MUL')

	def __truediv__(self, other):
		return Operation(self, other, 'DIV')

	def __rtruediv__(self, other):
		return Operation(other, self, 'DIV')

	def __pow__(self, exp):
		return Operation(self, exp, 'POW')

	def __rpow__(self, base):
		return Operation(base, self, 'POW')

	def __neg__(self):
		return Operation(self, -1.0, 'MUL')

	def __float__(self):
		if self.op == 'ADD':
			return float(self.v1) + float(self.v2)
		elif self.op == 'SUB':
			return float(self.v1) - float(self.v2)
		elif self.op == 'MUL':
			return float(self.v1) * float(self.v2)
		elif self.op == 'DIV':
			return float(self.v1) / float(self.v2)
		elif self.op == 'POW':
			return float(self.v1) ** float(self.v2)
		elif self.op == 'VAR':
			return float(self.v1)
		else:
			raise NotImplementedError(f'Operation {self.op} undefined')

class Variable(Operation):
	def set(self, value):
		self.v1 = value
