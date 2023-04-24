import pandas as pd 

import numpy as np 

df= pd.read_csv('D:/vs/yolo/iris.csv')
#print(df.head())
#print('abcdefgh')
#print(df.columns)
#print(df.shape)
#print(df.isnull().sum())
#print(df.duplicated().sum())
df.drop_duplicates(inplace= True)
#print(df.duplicated().sum())
#print(df.dtypes)

#print(df['label'].value_counts())
x= df.drop('label',axis= 1)
y= df['label']
#print(x.shape)
#print(y.shape)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.30,random_state= 50)
#print(xtrain.shape)
#print(ytrain.shape)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

rf = RandomForestClassifier(criterion= 'gini', n_estimators= 80, max_depth=4,min_samples_split=15)
lr = LogisticRegression()
kn= KNeighborsClassifier(n_neighbors=12)

kn.fit(xtrain,ytrain)
lr.fit(xtrain,ytrain)
rf.fit(xtrain,ytrain)


'''print('test scorelr',lr.score(xtest,ytest))
print('train score lr',lr.score(xtrain,ytrain))

print('test score kn',kn.score(xtest,ytest))
print('train score kn',kn.score(xtrain,ytrain))

print('test score rf',rf.score(xtest,ytest))
print('train score rf',rf.score(xtrain,ytrain))'''


import pickle
pickle.dump(rf,open('rf_model.pkl','wb'))
pickle.dump(lr,open('lr_model.pkl','wb'))
pickle.dump(kn,open('kn_model.pkl','wb'))
