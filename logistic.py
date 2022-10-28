import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('bc.csv')
df

def getCols(df):
    l=LabelEncoder()
    df['diagnosis']=l.fit_transform(df['diagnosis'])
    X=df.drop(['id','diagnosis'],axis=1)
    df=df.drop('id',axis=1)
    Y=df['diagnosis']
    return X,Y
X,Y=getCols(df)

X

pd.DataFrame(Y)

plt.figure(figsize=(20,20))
sns.heatmap(X.corr(),annot=True)

def remove_multicollinearity(X):
    ind = X.columns
    final_features = list(ind)
    p=df[X.columns].corr().values.tolist()
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            if(abs(p[i][j])>0.7 and ind[i] in final_features):
                final_features.remove(ind[i])
    print("Before: ",ind)
    print("After: ",final_features)
    return final_features
ind = X.columns
features = remove_multicollinearity(X)

for i in ind:
    if i not in features:
        X=X.drop(i,axis=1)
        df=df.drop(i,axis=1)
X

plt.figure(figsize=(20,20))
for i in range(len(features)):
    plt.subplot(4,3,i+1)
    sns.boxplot(y=features[i],x='diagnosis',data=df)
plt.show()

df=df.drop('id',axis=1)
def outlier_treatment(df, feature):
    q1, q3 = np.percentile(df[feature], [25, 75])
    IQR = q3 - q1 
    lower_range = q1 - (1.5 * IQR) 
    upper_range = q3 + (1.5 * IQR)
    to_drop = df[(df[feature]<lower_range)|(df[feature]>upper_range)]
    df.drop(to_drop.index, inplace=True)
for i in features:
    outlier_treatment(df, i)
df

plt.figure(figsize=(20,20))
for i in range(len(features)):
    plt.subplot(4,3,i+1)
    sns.boxplot(y=features[i],x='diagnosis',data=df)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

from sklearn.linear_model import LogisticRegression
l = LogisticRegression()
l.fit(X_train,y_train)
y_pred=l.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
TP=cm[0][0]
FP=cm[1][0]
FN=cm[0][1]
TN=cm[1][1]
cm

P=TP+FN
N=TN+FP
print("accuracy: ",(TP+TN)/(TP+FP+TN+FN))
print("precision for p: ",TP/(TP+FP))
print("precision for n: ",TN/(TN+FN))
print("recall for p/TPR: ",TP/(TP+FN))
print("recall for n/TNR: ",TN/(FP+TN))
print("FPR: ",FP/N)
print("FNR: ",FN/P)