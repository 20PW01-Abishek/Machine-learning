from sklearn import datasets
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
dataset = datasets.load_iris()
X = df = pd.DataFrame(dataset['data'],columns=dataset['feature_names'])
def PCA(X,k):
    df = X.copy()
    df = df-df.mean()
    cv = list(df.cov().values)
    w, v = np.linalg.eig(cv)
    ind = np.argsort(w)[::-1]
    w = w[ind]
    v = v[:,ind]
    
    W = v[:k].T
    Z = X.dot(W)
    Z.columns = [str(i) for i in range(1,k+1)]
    return Z
y=dataset.target
print(PCA(X,2))
print(dataset.target)
print(X)
