import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

X, y = make_classification(n_samples=100, n_features=5, n_classes=2)
y[y==0] = -1

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, train_size=80)

plt.scatter(X[:, 0], X[:, 1], c=y, label='Overall Dataset') 
plt.title("Scatter plot of Overall Dataset")
plt.show()

plt.scatter(X_train[:, 0], X_train[:, 1], c='r')
plt.scatter(X_test[:, 0], X_test[:, 1], c='b')
plt.legend(['Training Dataset', 'Testing Dataset'])
plt.title('Scatter plot of Training, and Testing Dataset')
plt.show()

class MyPerceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.lr = learning_rate
        self.epochs = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
    
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.activation_function(np.dot(self.weights, X[i]) + self.bias)
                self.weights += self.lr * (y[i] - y_pred) * X[i]
                self.bias += self.lr * (y[i] - y_pred)
        
        print("Weight Vector:", self.weights)
        return self.weights, self.bias

    def activation_function(self, activation):
        if activation >= 0:
            return 1
        else: 
            return -1

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]): 
            y_pred.append(self.activation_function(np.dot(self.weights, X[i]) + self.bias))
        return y_pred

learning_rate = 0.01
print('Learning Rate:', learning_rate)
perceptron = MyPerceptron(learning_rate)
weights, bias = perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
mis_classify = np.sum(y_pred != y_test)

print('No. Of Misclassification:', mis_classify)
print('Accuracy:', accuracy_score(y_test, y_pred) * 100, '%')

x0_1 = np.amin(X_test[:, 0])
x0_2 = np.amax(X_test[:, 0])
x1_1 = (-weights[0] * x0_1 - bias) / weights[1]
x1_2 = (-weights[0] * x0_2 - bias) / weights[1]

plt.plot([x0_1, x0_2], [x1_1, x1_2])
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', s=20)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x') 
plt.legend(['Decision Boundary', 'Testing Dataset', 'Predicted Dataset']) 
plt.title('Scatter plot of Testing and Predicted Dataset for learning rate: {0}'.format(learning_rate))
plt.show()

cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[-1, 1])
cm_display.plot()
plt.title("Confusion Matrix of Learning Rate = {0}".format(learning_rate))
plt.show()
