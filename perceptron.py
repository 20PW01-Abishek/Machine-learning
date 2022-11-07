import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.activation_func = self.unitStepFunction
        self.w = None
        self.b = None

    def fit(self, X, y):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.iterations):
            for ii, x_i in enumerate(X):
                y_value = np.dot(x_i, self.w) + self.b
                y_pred = self.activation_func(y_value)
                updateVal = self.lr * (y_[ii] - y_pred)
                self.w += updateVal * x_i
                self.b += updateVal

    def predict(self, X):
        y_value = np.dot(X, self.w) + self.b
        y_pred = self.activation_func(y_value)
        return y_pred

    def unitStepFunction(self, x):
        return np.where(x >= 0, 1, 0)
        


if __name__ == "__main__":
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    p = Perceptron(learning_rate=0.1, iterations=5000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.w[0] * x0_1 - p.b) / p.w[1]
    x1_2 = (-p.w[0] * x0_2 - p.b) / p.w[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()