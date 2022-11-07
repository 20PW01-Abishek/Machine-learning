import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def unitStepFunction(X):
    return np.where(X>=0,1,0)

class Perceptron:
    def __init__(self, eta, itr):
        self.eta = eta
        self.itr = itr
        self.activationFunction = unitStepFunction
        self.w = None
        self.b = None
        
    def fit(self,X,y):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0
        y = np.array([1 if i > 0 else 0 for i in y])
        
        for _ in range(self.itr):
            for i,X_i in enumerate(X):
                y_value = np.dot(X_i, self.w) + self.b
                y_pred = self.activationFunction(y_value)
                grad = self.eta * (y[i] - y_pred)
                self.w += grad * X_i
                self.b += grad
    
    def predict(self, X):
        y_value = np.dot(X, self.w) + self.b
        y_pred = self.activationFunction(y_value)
        return y_pred

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

p = Perceptron(0.01, 1000)
p.fit(X_train, y_train)
predictions = p.predict(X_test)
fig = plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)
