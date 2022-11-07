import pandas as pd
import numpy as np
import math

df=pd.read_csv('dataset.csv')
cols=list(df.columns)
t=[]
for i in range(len(df)):
    for j in range(len(df)):
        l=[]
        for k in cols:
            l.append((df[k][i]-df[k][j])**2)
        t.append(math.sqrt(sum(l)))
t=[round(i,2) for i in t]
A=[]
l=[]
for i in range(len(t)):
    l.append(t[i])
    if i%len(df)==len(df)-1:
        A.append(l)
        l=[]
        
D = [[0 for col in range(len(df))] for row in range(len(df))]
for i in range(len(D)):
    D[i][i]=len(D)-1

D=np.array(D)
A=np.array(A)
L=D-A

w, v = np.linalg.eig(L)
print(w)

i=0
for i in range(len(w)):
    if i >= 0:
        break
print(i)
f=v[i]
f=list(f)

for i in range(len(f)):
    if f[i]>=0:
        f[i]=1
    else:
        f[i]=0
print(f)

# Using sklearn

from sklearn.cluster import SpectralClustering
clustering = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0).fit(df)
print(list(clustering.labels_))