from sklearn import datasets
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def distance (x,y):
    dist=0
    for i in range(len(x)):
        dist+=np.square(x[i]-y[i])
    return np.sqrt(dist)

def kmeans(data, k):
    n=len(data[0])
    new_cluster=old_cluster = [data[random.randint(0, len(data)-1)] for i in range(k)]
    
    while True:
        labels=[]
        count=np.zeros(k)
        for i in range(len(data)):
            dist=[]
            for j in range(k):
                dist.append(distance(data[i],old_cluster[j]))
            temp=dist.index(min(dist))
            labels.append(temp)
            count[temp]+=1
            
        for i in range(k):
            new_cluster[i]=np.zeros(n)
        
        for i in range(len(data)):
            new_cluster[labels[i]] += data[i]/count[labels[i]]
            
        if np.array_equiv(old_cluster, new_cluster):
            break
        else:
            old_cluster = list(new_cluster)
        
    return {"labels":labels, "old": old_cluster}
        

iris = datasets.load_iris()
data = iris.data

inertias = []
for i in range(1, 11):
    model = KMeans(n_clusters=i)
    model.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

print(kmeans(data,3))