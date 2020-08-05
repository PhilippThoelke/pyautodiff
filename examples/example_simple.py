from pyautodiff import Variable

# create variables
v1 = Variable(3)
v2 = Variable(5)

# create the computation
comp1 = 10 * v1
comp2 = v1 ** 3
comp3 = v1 * v2
comp4 = ((v1 ** 8) / (10 * v2 ** 4)) + 3

# get the results and gradients
print(f'10 * v1 = {float(comp1)} | gradient: [v1: {comp1.gradient(v1)}]')
print(f'v1 ** 3 = {float(comp2)} | gradient: [v1: {comp2.gradient(v1)}]')
print(f'v1 * v2 = {float(comp3)} | gradient: [v1: {comp3.gradient(v1)}, v2: {comp3.gradient(v2)}]')
print(f'((v1 ** 8) / (10 * v2 ** 4)) + 3 = {float(comp4)} | gradient: [v1: {comp4.gradient(v1)}, v2: {comp4.gradient(v2)}]')
