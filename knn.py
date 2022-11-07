from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

iris = pd.read_csv('Iris.csv')
# iris = datasets.load_iris()
cols=iris.columns.tolist()
X = iris[cols[:-1]]
Y = iris[cols[-1]].to_numpy()

l=LabelEncoder()
Y=l.fit_transform(Y)

X = (X-X.mean())/X.std()
X = X.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

def distance (x,y):
    dist=0
    for i in range(len(x)):
        dist+=np.square(x[i]-y[i])
    return np.sqrt(dist)

def knn(X,Y,x,y,k):
    dist=[]
    for j in range(len(X)):
        dist.append([distance(X[j],x),Y[j]])
    dist.sort()
    k_dist = dist[:k]
    count=[0]*len(list(set(Y)))
    for n in k_dist:
        count[n[1]]+=1
    return count.index(max(count))
    
y_pred=[]
k=int(np.sqrt(len(y_train)))
for i in range(len(X_test)):
    y_pred.append(knn(X_train,y_train,X_test[i],y_test[i],k))

print(y_pred)
print(list(y_test))
ct = pd.crosstab(y_test, y_pred)
print(ct)