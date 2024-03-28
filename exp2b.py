import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def func_f(x, y):
    f = 10 * x**2 + 5 * x * y + 10 * (y - 3)**2
    return f

def df_dx(x, y):
    df = 5 * (4 * x + y)
    return df

def df_dy(x, y):
    df = 5 * (x + 4 * (y - 3))
    return df

stepsize = 0.0001

def gradient_optimization(initial_x, initial_y, stepsize, stopping_threshold):
    x = initial_x
    y = initial_y
    count = 0
    while(True):
        gradient_x = df_dx(x, y)  # calculate gradient wrt x
        gradient_y = df_dy(x, y)  # calculate gradient wrt y
        xdash = x - stepsize * gradient_x
        ydash = y - stepsize * gradient_y
        x = xdash
        y = ydash
        count += 1
        if count > 1 and np.linalg.norm([gradient_x, gradient_y]) < stopping_threshold:
            break
    return x, y, count

stepsize_array = [0.001, 0.005, 0.01, 0.05]
initialGuess = [10, 15]
minima_x = []
minima_y = []
iteration = []
funcValue_atMinima = []

for i in range(4):
    a, b, c = gradient_optimization(initialGuess[0], initialGuess[1], stepsize_array[i], 0.001)
    minima_x.append(a)
    minima_y.append(b)
    iteration.append(c)
    funcValue_atMinima.append(func_f(minima_x[i], minima_y[i]))

data = {'Minima value of x1': minima_x, 'Minima value of x2': minima_y, 'No of iterations': iteration, 'Func Value at minima': funcValue_atMinima}
df = pd.DataFrame(data)

print("Result of Gradient descent on bivariate function")
print(df)

x = np.linspace(-10, 10, 100)
y = np.linspace(-15, 15, 100)

X, Y = np.meshgrid(x, y)
Z = func_f(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Surface Plot of f(x1, x2)')
plt.colorbar(surf)
plt.show()
