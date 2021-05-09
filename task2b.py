# Put task2b.py code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

life = pd.read_csv('life.csv', encoding='ISO-8859-1')
world = pd.read_csv('world.csv', encoding='ISO-8859-1')

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
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#transfer the X_train into dataframe with column names to do feature engineering
X_train_data = pd.DataFrame(X_train,columns=data.columns)
X_test_data= pd.DataFrame(X_test,columns=data.columns)

print("Total number of countries after merging", X_train_data.shape[0])
print("Total numnber of features after merging", X_train_data.shape[1])

##feature engineering method 1: interaction term pairs
X_train_eng = X_train_data.copy()
X_test_eng = X_test_data.copy()

def create_interaction(df,var1,var2):
    name = var1 + "*" + var2
    df[name] = pd.Series(df[var1] * df[var2], name=name)

column = 20
for i in range(column):
    for j in range(i+1,column):
        create_interaction(X_train_eng, X_train_eng.columns[i],X_train_eng.columns[j])
        create_interaction(X_test_eng, X_test_eng.columns[i],X_test_eng.columns[j])

print("Training data set after adding 190 interaction term pairs:")
print(X_train_eng)

##feature engineering method 2: clustering labels
##decide k using elbow method 
from sklearn.cluster import KMeans
#use SSE
SSE = []  
for k in range(1, 11):
    model = KMeans(n_clusters=k)  
    model.fit(X_train)
    SSE.append(model.inertia_)
X = range(1, 11)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()
plt.savefig('task2bgraph1.png')
print("SSE of different k")
print(SSE)
print("After using elbow method, choose k=3 applied to k-means to do the clustering label.")

#add clustering label  (training)
model = KMeans(n_clusters=3)
model.fit(X_train_eng)
X_train_eng['clustering label'] = pd.Series(model.labels_)

#add clustering label (testing)
model = KMeans(n_clusters=3)
model.fit(X_test_eng)
X_test_eng['clustering label'] = pd.Series(model.labels_)

print("Training data of 211 features after feature generation")
print(X_train_eng)

##feature slection method1: Mutual Information
from sklearn.feature_selection import mutual_info_classif

NMI = {}
NMI_arr = mutual_info_classif(X_train_eng, y_train)
for i in range(X_train_eng.shape[1]):
    NMI[i] = NMI_arr[i]
NMI_4=sorted(NMI.items(),key=lambda x:x[1],reverse = True)[0:4]

print("Array stores top4 highest standard mutual information:")
print(NMI_4)

#use the top 4 feature to perform 3-NN classification
#train the classifiers
knn1 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_train_eng.iloc[:, [NMI_4[0][0],NMI_4[1][0],NMI_4[2][0],NMI_4[3][0]]], y_train)
#test the classifiers
y_pred=knn1.predict(X_test_eng.iloc[:, [NMI_4[0][0],NMI_4[1][0],NMI_4[2][0],NMI_4[3][0]]])
print("Accuracy of feature engineering: %.3f"%accuracy_score(y_test, y_pred))

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=4)

#PCA for training set
fit_x = pca.fit(X_train_data)
X_pca_train = fit_x.transform(X_train_data)

#PCA for testing set
X_pca_test = fit_x.transform(X_test_data)

print("Training data set after PCA:")
print(X_pca_train)

#use pca feature to perform 3-NN classification
#train the classifiers
knn2 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn2.fit(X_pca_train, y_train)
#test the classifiers
y_pred=knn2.predict(X_pca_test)
print("Accuracy of PCA: %.3f"%accuracy_score(y_test, y_pred))

#use the first 4 features to perform 3-NN classification
#train the classifiers
knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train_data.iloc[:,0:4], y_train)
#test the classifiers
y_pred=knn3.predict(X_test_data.iloc[:,0:4])
print("Accuracy of first four features: %.3f"%accuracy_score(y_test, y_pred))