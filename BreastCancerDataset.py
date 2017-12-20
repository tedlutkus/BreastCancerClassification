# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:44:01 2017

@author: harveylab
"""

#%% 
#In this script, I investigate and try to predict benign vs malignant
#cancer cells in a dataset from Wisconsin

#Import relevant modules for data preparation and visualization
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

#Machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#Import modules for statistical analysis
from scipy.stats import norm
from scipy.stats import ttest_ind
from scipy.stats import bartlett

df = pd.read_csv('C:/Users/harveylab/Documents/Python Scripts/Breast_Cancer_Dataset.csv')

#First I will examine the general attributes of the data and identify any issues that
#could come up later on
df.info()
df.dtypes
df.isnull().any()

#It appears that there is a column 'Unnamed: 32' that contains only NaN values so I will just drop this
df = df.drop(['Unnamed: 32'],axis=1)

#I also will convert the 'diagnosis' column from objects to categorical data
df['diagnosis'] = df['diagnosis'].astype('category')

#%% Data Exploration
#Peaking at data and also check how many M and B values we have
df.head()
df['diagnosis'].value_counts() #357 B and 212 M

#Comparing M and B features by mean value
df.groupby(['diagnosis']).mean() #It seems like malignant cells tend to be larger in size/area
                                #and have more sporadic geometry but this is just speculation
                                                                
#Now I will check the distributions of data and correlations between different variables.
#This will give some intuition as to what underlying structure may be in the data,
#which features are relevant, and whether or not feature engineering may be needed.
#I start by defining a new dataframe with only the 'mean' features because the 'se' and 'worst' features will be redundant for correlations                  
df_mean = df[['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean',
                       'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']]
df_mean_numeric = df_mean.drop(['diagnosis'],axis=1)
corr_matrix = df_mean_numeric.corr()
sns.heatmap(corr_matrix,vmin=-1,vmax=1,center=0,annot=True,fmt='.2f')
plt.title('Correlations of Numeric Data')
#What stands out to me from the correlation matrix is that radius, perimeter, and area are so strongly correlated that
#they could be engineered into a feature called 'size'. It is also interesting that there is almost no relationship between smoothness and texture

#Check distributions of data
sns.pairplot(data=df_mean,hue='diagnosis',vars=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean',
                                           'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean'])
plt.title('Relationships Between Features')
#Much of the data seems to follow a Weibull distribution where the PDF peaks early and decays slowly

#Plot the relationships between radius, perimeter, and area (RPA) more in depth
df_RPA = df[['diagnosis','radius_mean','perimeter_mean','area_mean']]
sns.pairplot(df_RPA,hue='diagnosis')
plt.title('Relationships Between Size-Dependant Features')

#%% Dimensionality Reduction of Radius, Perimeter, and Area
#Due to the near perfect correlation between radius, perimeter, and area, I believe that these features can be described by one term
#While Principal Component Analysis (PCA) is typically used for larger dimensionality reduction,
#I will use it here to find projections of the three features onto a one dimensional vector.
#At the end of this analysis, I will compare models with and without PCA to see whether or not there was improvement in prediction accuracy

#Create a numpy array
arr_PCA = df[['radius_mean','perimeter_mean','area_mean']].values

#Scale the values
arr_PCA = scale(arr_PCA)

#Fitting
pca = PCA(n_components=3)
pca.fit(arr_PCA)

#Calculate the variance explained by each reduced dimension
var = pca.explained_variance_ratio_

#Cumulative variance calculated and plotted
var_c = np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)
plt.plot([1,2,3],var_c)
plt.xticks([1,2,3])
plt.ylim([98,100])
plt.title('Principal Component Analysis of Size Features')
plt.ylabel('Variance of Original Data Explained (%)')
plt.xlabel('Principal Components')

#I choose to stick with one principal component
pca1 = PCA(n_components=1)
arr_PCA1 = pca1.fit_transform(arr_PCA)

#Save the dataframe without PCA for later comparison in model prediction accuracy
df_mean_noPCA = df_mean

#Replace the old feature with the new 'size' feature
df_mean_new = df_mean.drop(['radius_mean','perimeter_mean','area_mean'],axis=1)
df_mean_new['size'] = arr_PCA1

#I recalculate the correlation coefficients with the new feature and plot a heatmap, it looks like the new size feature
#retains the variance that was seen with the three features previously
corr_matrix1 = df_mean_new.corr()
sns.heatmap(corr_matrix1,vmin=-1,vmax=1,center=0,annot=True,fmt='.2f')
plt.title('Correlations of New Features')

#%% Model Training and Selection

#Problem type: Classification (Malignant or Benign)

#Models:
#1. Logistic Regression
#2. Support Vector Classifier
#3. K-Nearest Neighbor
#4. Gaussian Naive Bayes
#5. Random Forest

#Replacing the M and B strings in 'diagnosis' with binary values (M=1 B=0)
df_mean_new['diagnosis'] = df_mean_new['diagnosis'].map({'B':0,'M':1}).astype(int)
df_mean_noPCA['diagnosis'] = df_mean_new['diagnosis']

#Splitting input and target data
X = df_mean_new.drop(['diagnosis'],axis=1)
y = df_mean_new['diagnosis']
X_noPCA = df_mean_noPCA.drop(['diagnosis'],axis=1)
y_noPCA = y

#Run through 100 trained models with a new random state on each pass
acc = []
classifier = []
PCA_Ind = []

for i in range(0,100):
    #Setting up training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=i)
    X_train_noPCA, X_test_noPCA, y_train_noPCA, y_test_noPCA = train_test_split(X_noPCA,y_noPCA,random_state=i)
    
    #Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)
    y_pred = logreg.predict(X_test)
    acc.append(logreg.score(X_train,y_train)*100)
    classifier.append('Logistic Regression')
    PCA_Ind.append('Yes')
    
    #Logistic Regression w/o PCA
    logreg.fit(X_train_noPCA,y_train_noPCA)
    y_pred_noPCA = logreg.predict(X_test_noPCA)
    acc.append(logreg.score(X_train_noPCA,y_train_noPCA)*100)
    classifier.append('Logistic Regression')
    PCA_Ind.append('No')
    
    #Support Vector Classifier
    svc = SVC()
    svc.fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    acc.append(svc.score(X_train,y_train)*100)
    classifier.append('Support Vector Classifier')
    PCA_Ind.append('Yes')
    
    #Support Vector Classifier w/o PCA
    svc.fit(X_train_noPCA,y_train_noPCA)
    y_pred_noPCA = svc.predict(X_test_noPCA)
    acc.append(svc.score(X_train_noPCA,y_train_noPCA)*100)
    classifier.append('Support Vector Classifier')
    PCA_Ind.append('No')
    
    #K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    acc.append(knn.score(X_train,y_train)*100)
    classifier.append('K-Nearest Neighbors')
    PCA_Ind.append('Yes')
    
    #K-Nearest Neighbors w/o PCA
    knn.fit(X_train_noPCA,y_train_noPCA)
    y_pred_noPCA = knn.predict(X_test_noPCA)
    acc.append(knn.score(X_train_noPCA,y_train_noPCA)*100)
    classifier.append('K-Nearest Neighbors')
    PCA_Ind.append('No')
    
    #Gaussian Naive Bayes
    gaussian = GaussianNB()
    gaussian.fit(X_train,y_train)
    y_pred = gaussian.predict(X_test)
    acc.append(gaussian.score(X_train,y_train)*100)
    classifier.append('Gaussian Naive Bayes')
    PCA_Ind.append('Yes')
    
    #Gaussian Naive Bayes w/o PCA
    gaussian.fit(X_train_noPCA,y_train_noPCA)
    y_pred_noPCA = gaussian.predict(X_test_noPCA)
    acc.append(gaussian.score(X_train_noPCA,y_train_noPCA)*100)
    classifier.append('Gaussian Naive Bayes')
    PCA_Ind.append('No')
    
    #Random Forest
    random_forest = RandomForestClassifier(n_estimators=10)
    random_forest.fit(X_train,y_train)
    y_pred = random_forest.predict(X_test)
    acc.append(random_forest.score(X_train,y_train)*100)
    classifier.append('Random Forest')
    PCA_Ind.append('Yes')
    
    #Random Forest w/o PCA
    random_forest.fit(X_train_noPCA,y_train_noPCA)
    y_pred_noPCA = random_forest.predict(X_test_noPCA)
    acc.append(random_forest.score(X_train_noPCA,y_train_noPCA)*100)
    classifier.append('Random Forest')
    PCA_Ind.append('No')


#%% Results
#Now that the models are trained and I have results, I will compare the different models
#As well as assess the implementation of PCA

#Setup lists and assign columns in dataframe
data = {'Classifier':classifier,
        'PCA':PCA_Ind,
        'Prediction Accuracy':acc}
df_results = pd.DataFrame(data)
df_results = df_results.sort_values('Prediction Accuracy',ascending=False)

sns.set_style('darkgrid')
sns.boxplot(x='Classifier',y='Prediction Accuracy',data=df_results,hue='PCA')
plt.title('Classifier Performance on Test Data')
plt.ylabel('Prediction Accuracy (%)')
plt.xlabel('Classifier')
plt.ylim([88,100])
#It is evident that Random Forest performed best on this dataset so I will now run a two sample T-test
#To determine if the usage of PCA resulted in a significant performance change

#%% Two Sample T-Test on Random Forest Classifier

#Format arrays of prediction accuracy for PCA vs w/o PCA
df_random_forest = df_results.loc[(df_results['Classifier']=='Random Forest') & (df_results['PCA']=='Yes')]
df_random_forest_noPCA = df_results.loc[(df_results['Classifier']=='Random Forest') & (df_results['PCA']=='No')]
arr_random_forest = df_random_forest['Prediction Accuracy'].values
arr_random_forest_noPCA = df_random_forest_noPCA['Prediction Accuracy'].values

#Check the distribution of data
plt.subplot(2,1,1)
sns.distplot(arr_random_forest,fit=norm)
plt.title('Random Forest')
plt.ylabel('Frequency')
plt.xlabel('Prediction Accuracy (%)')
plt.subplot(2,1,2)
sns.distplot(arr_random_forest_noPCA,fit=norm)
plt.title('Random Forest No PCA')
plt.ylabel('Frequency')
plt.xlabel('Prediction Accuracy (%)')
#I will assume that the data is normally distributed

#Perform Bartlett's test for equal variances
_,pval_variance = bartlett(arr_random_forest,arr_random_forest_noPCA)
print(pval_variance)
#p-value > 0.05 therefore we fail to reject the null hypothesis that the samples have equal variances

#Perform two sample T-test for equal means
_,pval_ttest = ttest_ind(arr_random_forest,arr_random_forest_noPCA)
print(pval_ttest)
#p-value > 0.05 so again we fail to reject the null hypothesis, suggesting that there is no significant change in model performance
#with the addition of PCA

#%% Conclusion
#Consider LDA