import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Step 1: Generate make_regression dataset
x, y = make_regression(n_samples=100, n_features=1, bias=0, noise=5)

# Scale feature x to range -5…..5 and target y to range 15…..-15
x_scaled = np.interp(x, (x.min(), x.max()), (-5, 5))
y_scaled = np.interp(y, (y.min(), y.max()), (15, -15))

# Step 2: Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Step 3: Add bias to training dataset
x_train_b = np.c_[np.ones((len(x_train), 1)), x_train]

# Step 4: Obtain linear regression coefficients using closed-form solution
w_b = np.linalg.inv(x_train_b.T.dot(x_train_b)).dot(x_train_b.T).dot(y_train)

# Step 5: Test the algorithm using testing dataset and obtain predictions
x_test_b = np.c_[np.ones((len(x_test), 1)), x_test]
y_pred = x_test_b.dot(w_b)

# Step 6: Calculate RMSE for training and testing
rmse_train = np.sqrt(mean_squared_error(y_train, x_train_b.dot(w_b)))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

# Step 7: Plot scatter plots with regression lines
plt.scatter(x_scaled, y_scaled, label="Overall Dataset")
plt.scatter(x_train, y_train, label="Training Dataset")
plt.scatter(x_test, y_test, label="Testing Dataset")
plt.plot(x_scaled, x_scaled*w_b[1] + w_b[0], color='red', label='Regression Line')
plt.xlabel('Feature (x)')
plt.ylabel('Target (y)')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Step 8: Verify results using scikit-learn's LinearRegression
lr = LinearRegression()
lr.fit(x_train.reshape(-1, 1), y_train)
y_pred_sklearn = lr.predict(x_test.reshape(-1, 1))

# Check if the coefficients are similar
if np.allclose(lr.coef_, w_b[1]) and np.allclose(lr.intercept_, w_b[0]):
    print("Results are verified.")
else:
    print("Results are not verified.")

print("RMSE (Training):", rmse_train)
print("RMSE (Testing):", rmse_test)
