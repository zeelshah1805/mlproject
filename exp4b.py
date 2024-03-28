from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate make_regression dataset
x, y = make_regression(n_samples=100, n_features=1, bias=0, noise=5, random_state=42)

# Step 2: Scale feature x to range -5…..5 and target y to range 15…..-15
x_scaled = np.interp(x, (x.min(), x.max()), (-5, 5))
y_scaled = np.interp(y, (y.min(), y.max()), (15, -15))

# Step 3: Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Step 4: Add bias to training data
x_train_b = np.c_[np.ones((len(x_train), 1)), x_train]

# Step 5: Use Ridge Regression instead of Linear Regression
ridge_reg = Ridge(alpha=1.0)  # You can adjust the alpha parameter here
ridge_reg.fit(x_train_b, y_train)

# Step 6: Test the algorithm using testing dataset and obtain predictions
x_test_b = np.c_[np.ones((len(x_test), 1)), x_test]
y_pred = ridge_reg.predict(x_test_b)

# Step 7: Find and display Root Mean Square Error (RMSE) for training and testing
rmse_train = np.sqrt(mean_squared_error(y_train, ridge_reg.predict(x_train_b)))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE - Training:", rmse_train)
print("RMSE - Testing:", rmse_test)

# Step 8: Plot scatter plot with Ridge Regression line
plt.scatter(x_scaled, y_scaled, label='Overall Dataset')
plt.scatter(x_train, y_train, color='red', label='Training Dataset')
plt.scatter(x_test, y_test, color='green', label='Testing Dataset')
plt.plot(x_test, y_pred, color='blue', label='Ridge Regression Line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()

# Step 9: Verify results with sklearn's Ridge Regression
print("Ridge Regression Coefficients:", ridge_reg.coef_)
print("Ridge Regression Intercept:", ridge_reg.intercept_)
print("Ridge Regression RMSE - Testing:", rmse_test)
