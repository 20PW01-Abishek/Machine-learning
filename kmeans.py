from sklearn import datasets
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def distance(x,y):
    d=0
    for i in range(len(x)):
        d+=(x[i]-y[i])**2
    return np.sqrt(d)
    
def kmeans(data,k):
    n=len(data[0])
    old_centroid = new_centroid = [data[random.randint(0, len(data)-1)] for i in range(k)]
    while True:
        labels=[]
        count=np.zeros(k)
        for i in range(len(data)):
            dist=[]
            for j in range(k):
                dist.append(distance(data[i],old_centroid[j]))
            temp=dist.index(min(dist))
            labels.append(temp)
            count[temp]+=1
            
        for i in range(k):
            new_centroid[i]=np.zeros(n)
        
        for i in range(len(data)):
            new_centroid[labels[i]] += data[i]/count[labels[i]]
            
        if np.array_equiv(old_centroid, new_centroid):
            break
        else:
            old_centroid = list(new_centroid)
        
    return {"labels":labels, "old": old_centroid}
        

iris = datasets.load_iris()
data = iris.data

inertias=[]
for i in range(1,21):
    model = KMeans(n_clusters=i)
    model.fit_transform(data)
    inertias.append(model.inertia_)

plt.plot(range(1,21),inertias,marker='o')

y_pred=kmeans(data,3)
print(y_pred)
