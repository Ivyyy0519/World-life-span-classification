# Put task2a.py code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

life = pd.read_csv('life.csv', encoding='ISO-8859-1')
world = pd.read_csv('world.csv', encoding='ISO-8859-1')

#prepare the dataset and discard any countries not present in both csv
total = pd.merge(life, world, on='Country Code',how = 'inner')
total = total.sort_values(by='Country Code',ascending=True)
total.replace( '..', np.nan,inplace = True)

##get just the features
data=total.iloc[:,6:27].astype(float)

##get just the class labels
classlabel=total['Life expectancy at birth (years)']

#split data and data pre-processing
##randomly select 70% of the instances to be training and the rest to be testing
X_train, X_test, y_train, y_test = train_test_split(data, classlabel, train_size=0.7, test_size=0.3, random_state=200)

#median imputation to missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

#normalise the data to have 0 mean and unit variance 
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s=scaler.transform(X_train)
X_test_s=scaler.transform(X_test)

#decision tree
#train the classifiers
dt = DecisionTreeClassifier(random_state=200, max_depth=3)
dt.fit(X_train_s, y_train)

#test the classifiers
y_pred=dt.predict(X_test_s)
print("Accuracy of decision tree: %.3f"%accuracy_score(y_test, y_pred))

#knn n=3
#train the classifiers
knn1 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_train_s, y_train)

#test the classifiers
y_pred1=knn1.predict(X_test_s)
print("Accuracy of k-nn (k=3): %.3f"%accuracy_score(y_test, y_pred1))

#knn n=7
#train the classifiers
knn2 = neighbors.KNeighborsClassifier(n_neighbors=7)
knn2.fit(X_train_s, y_train)

#test the classifiers
y_pred2=knn2.predict(X_test_s)
print("Accuracy of k-nn (k=7): %.3f"%accuracy_score(y_test, y_pred2))

task2a = pd.DataFrame(columns=('feature', 'median', 'mean', 'variance'))
for i in range(X_train.shape[1]):
    new = pd.DataFrame({'feature':X_train.columns[i],'median':'%.3f' % X_train.iloc[:,i].median(),'mean':'%.3f' % X_train.iloc[:,i].mean(),'variance':'%.3f' % X_train.iloc[:,i].var()},index=[i])
    task2a = task2a.append(new,ignore_index=True)

task2a.to_csv("task2a.csv",index=False,sep=',',float_format='%.3f') 