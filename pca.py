from sklearn import datasets
import random
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
Y = iris.target

print(X)