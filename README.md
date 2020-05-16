# pyautodiff

A very small Python library (<80 LOC) capable of automatic differentiation of addition, subtraction, multiplication, division and potentiation. The library consists of a `Operation` class which stores a binary mathematical operation and a small subclass `Variable`, which only adds functionality for setting a variable's value. It supports having multiple variables in one calculation and computes the gradient using the `gradient` function. This function accepts either a single `Variable` object or an iterable of `Variable` objects. If a single object is passed, the function returns the derivative of the calculation defined in the `Operation` object with respect to the passed `Variable` object. If an iterable of `Variable` objects is passed to `gradient`, a list of partial derivatives with respect to the passed `Variable` objects is returned.

## Example
```python
from pyautodiff import Variable

# create variables
x = Variable(5)
y = Variable(2)

# create the computation
computation = (3 * x ** 2) / (y + 2)

# get the numerical value of the computation
result = float(computation)

# get the gradient (partial derivatives with respect to x and y)
gradient = computation.gradient((x, y))
```
