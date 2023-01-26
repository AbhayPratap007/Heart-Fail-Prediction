
#import libraries

import numpy as np # linear algebra
import pandas as pd # data processing
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


#read data

Heart_Fail=pd.read_csv('D:/Chanchal/heart_failure.csv')
Heart_Fail

#Columns

Heart_Fail.columns

#Data Visualization

sns.distplot(Heart_Fail.age)

sns.histplot(Heart_Fail.age)

sns.countplot(Heart_Fail.anaemia)

sns.distplot(Heart_Fail.creatinine_phosphokinase)

sns.histplot(Heart_Fail.creatinine_phosphokinase)

#Plots

sns.countplot(Heart_Fail.diabetes)

sns.displot(Heart_Fail.ejection_fraction)

sns.histplot(Heart_Fail.ejection_fraction)

sns.countplot(Heart_Fail.high_blood_pressure)

sns.histplot(Heart_Fail.platelets	)

sns.displot(Heart_Fail.platelets)

sns.histplot(Heart_Fail.serum_creatinine)

sns.displot(Heart_Fail.serum_creatinine)

sns.histplot(Heart_Fail.serum_sodium)

sns.displot(Heart_Fail.serum_sodium)

sns.countplot(Heart_Fail.sex)

sns.countplot(Heart_Fail.smoking)

sns.displot(Heart_Fail.time)

sns.histplot(Heart_Fail.time)

sns.countplot(Heart_Fail.DEATH_EVENT)


#It computes the pairwise correlation of columns, excluding NA/null values

Heart_Fail.corr()

#Information 
Heart_Fail.info()

#Data Plots
sns.boxplot(Heart_Fail.creatinine_phosphokinase)

sns.boxplot(Heart_Fail.ejection_fraction)

sns.boxplot(Heart_Fail.platelets)

sns.boxplot(Heart_Fail.serum_creatinine)

sns.boxplot(Heart_Fail.serum_sodium)

#Data Preprocessing
X = Heart_Fail.iloc[:,:-1].values
y = Heart_Fail.iloc[:,-1].values

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#X
#y

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#X_train
#y_train

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Accuracy
y_pred = classifier.predict(X_test)
log = accuracy_score(y_test,y_pred)*100
print("Accuracy Score:",accuracy_score(y_test,y_pred)*100,"%")
print("Recall Score:",recall_score(y_test,y_pred)*100,"%")
print("Precision Score:",precision_score(y_test,y_pred)*100,"%")
print("F1 Score:",f1_score(y_test,y_pred)*100,"%")
pd.crosstab(y_pred,y_test)

