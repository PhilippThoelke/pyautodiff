# pyautodiff

A very small Python library (<90 LOC) capable of automatic differentiation of addition, subtraction, multiplication, division and exponentiation. The library consists of a Operation class which stores a binary mathematical operation and a small subclass Variable, which only adds functionality for setting a variable's value. It supports having multiple variables in one calculation and computes the gradient using the gradient function. This function accepts either a single Variable object or an iterable of Variable objects. If a single object is passed, the function returns the derivative of the calculation defined in the Operation object with respect to the Variable object passed to the function. If an iterable of Variable objects is passed to gradient, a list of partial derivatives with respect to the variables in the iterable is returned.

The whole library is contained in `pyautodiff.py` and contains two classes. Only the Variable class is necessary for interfacing with the library and differentiating mathematical expressions. It is able to differentiate arbitrarily complex expressions containing addition, subtraction, multiplication, division and exponentiation with multiple variables.

`pyautodiff` is not intended to be used real world problems but as a learning resource. It lacks error handling in many cases and is very inefficient due to the implementation in pure Python code.

## C++ version
I have also implemented automatic differentiation in C++ (**WIP**). The code can be found [here](https://github.com/PhilippThoelke/autodiff).

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
More examples can be found in the following files:
- `example_simple.py`
  - simple calculations and differentiation with pyautodiff
- `example_perceptron.py`
  - implmementation of a perceptron with a sigmoid activation function
  - iterative training for classification of a logical AND using gradient descent
  - gradient computed using pyautodiff
- `example_mlp.py`
  - implementation of a multilayer perceptron (MLP) with variable layer sizes and sigmoid activation function
  - iterative training for classification of the non-linear logical XOR
  - MLP is trained with gradient descent using pyautodiff
