import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.linspace(-(np.sqrt(1.5)), 0, 100)

y = x * (np.exp(-(np.square(x))))  # banana function

plt.xlabel("x")
plt.ylabel("xe^-x^2")
plt.title("Banana Function")
plt.grid(True)
plt.plot(x, y)
plt.show()

def func_y_x(x):
    y = x * (np.exp(-(np.square(x))))  # banana function
    return y

def dy_dx(x):
    dy = (1 - 2 * (np.square(x))) * (np.exp(-(np.square(x))))
    return dy

stepsize = 0.0001

def gradient_optimization(initial_guess, stepsize, stopping_threshold):
    x = initial_guess
    count = 0
    while(True):
        gradient = dy_dx(x)  # calculate gradient of x
        xdash = x - stepsize * gradient
        x = xdash
        count += 1
        if count > 1 and np.linalg.norm(gradient) < stopping_threshold:
            break
    return x, count

stepsize_array = [0.001, 0.005, 0.01, 0.05]
initialGuess = -(np.sqrt(1.5))

minima = []
iteration = []
funcValue_atMinima = []

for i in range(4):
    a, b = gradient_optimization(initialGuess, stepsize_array[i], 0.001)
    minima.append(a)
    iteration.append(b)
    funcValue_atMinima.append(func_y_x(minima[i]))

data = {'Minima value': minima, 'No of iterations': iteration, 'Func Value at minima': funcValue_atMinima}
df = pd.DataFrame(data)

print("Result of Gradient descent on univariate function")
print(df)
