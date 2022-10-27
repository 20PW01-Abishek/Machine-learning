import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def getCols(df):
    l = LabelEncoder()
    l.fit(df['Class'])
    df['Class'] = l.transform(df['Class'])
    X = df.drop(['Sample code number', 'Class'], axis=1)
    Y = df['Class'].values
    return X,Y

df = pd.read_csv("./Dataset/bcoriginal.csv")
df = df.replace('?', np.nan)
df = df.apply(lambda x: x.fillna(str(int(x.median()))),axis=0)
X,Y=getCols(df)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

def prior_probability_count(Y):
    c1=c2=0
    for i in Y_train:
        if i==0:
            c1+=1
        else:
            c2+=1
    return c1,c2

c1,c2=prior_probability_count(Y_train)
print(c1," ",c2)

def construct_cpt(X,Y):
    unique=list(set(X))
    X=list(X)
    cpt={}
    for i in unique:
        l=[0,0]
        for j in range(len(X)):
            if i==X[j]:
                l[Y[j]]+=1
            cpt[i]=l
    print(cpt)
    return cpt

ind=X_train.columns
CPT={}
for i in ind:
    print("\nCPT for",i,":")
    CPT[i]=construct_cpt(X_train[i], Y_train)