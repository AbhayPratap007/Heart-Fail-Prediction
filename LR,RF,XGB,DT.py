#--------------Logistic regression, KNeaset Neighbours, Decision Tree, Random Forest, XGBoost-----------------


#Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context= "notebook", color_codes=True)

from scipy import stats
from scipy.stats import norm, skew, boxcox
from collections import Counter

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
import cufflinks as cf
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.offline as pyo
init_notebook_mode(connected=True)
cf.go_offline()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import warnings
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")

#Install Cufflinks
#pip install cufflinks 

#Data Preprocessing

Heart_Fail=pd.read_csv('D:/Chanchal/heart_failure.csv')
Heart_Fail

Heart_Fail.head()

Heart_Fail.shape

#Description
Heart_Fail.describe().T

#Missing values
Heart_Fail.isnull().sum()

Heart_Fail.nunique()

#Dividing features into numerical and Categorical

col = list(Heart_Fail.columns)
categorical_features = []
numerical_features = []
for i in col:
    if len(Heart_Fail[i].unique()) > 2:
        numerical_features.append(i)
    else:
        categorical_features.append(i)

print('Categorical Features :',*categorical_features)
print('Numerical Features :',*numerical_features)

Heart_Fail['age'] = Heart_Fail['age'].astype(int)
Heart_Fail['platelets'] = Heart_Fail['platelets'].astype(int)
Heart_Fail1 = Heart_Fail.copy(deep = True)

Heart_Fail1.head()

#Visualization
#Visualizing categorical Features

fig = make_subplots(
    rows=3, cols=2, subplot_titles=('<b>Distribution Of Anemia<b>','<b>Distribution Of Diabetes<b>','<b>Distribution Of High blood Pressure<b>',
                                   '<b>Distribution Of Sex<b>','<b>Distribution Of Smoking Status<b>', '<b>Distribution Of Death Event<b>'),
    vertical_spacing=0.01,
    specs=[[{"type": "pie"}       ,{"type": "pie"}] ,
           [{"type": "pie"}       ,{"type": "pie"}] ,
           [{"type": "pie"}       ,{"type": "pie"}] ]
           
)
fig.add_trace(
    go.Pie(values=Heart_Fail1.anaemia.value_counts().values,labels=['<b>Female<b>','<b>Male<b>','<b>Other<b>'],
           hole=0.3,pull=[0,0.08,0.2],marker_colors=['#dcd6f7','#a6b1e1'],textposition='inside'),
    row=1, col=1
)

fig.add_trace(
    go.Pie(values=Heart_Fail1.diabetes.value_counts().values,labels=['<b>1<b>','<b>0<b>'],
           
           hole=0.3,pull=[0,0.08,0.3],marker_colors=['#393E46', '#a696c8'],textposition='inside'),
    row=1, col=2
)


fig.add_trace(
    go.Pie(values=Heart_Fail.high_blood_pressure.value_counts().values,labels=['<b>1<b>','<b>0<b>'],
           hole=0.3,pull=[0,0.08,0.3],marker_colors=['#dcd6f7', '#424874'],textposition='inside'),
    row=2, col=1
)

fig.add_trace(
    go.Pie(values=Heart_Fail.sex.value_counts().values,labels=['<b>Yes<b>','<b>No<b>'],
           hole=0.3,pull=[0,0.08,0.3],marker_colors=['#a45fbe', '#a6b1e1'],textposition='inside'),
    row=2, col=2
)

fig.add_trace(
    go.Pie(values=Heart_Fail.smoking.value_counts().values,labels=['<b>Private<b>', '<b>Self-employed<b>', '<b>Govt_job<b>', '<b>children<b>', '<b>Never_worked<b>'],
           hole=0.3,pull=[0,0.08,0.08,0.08,0.2],marker_colors=['#77529e',  '#a5bdfd'],textposition='inside'),
    row=3, col=1
    
)
fig.add_trace(
    go.Pie(values=Heart_Fail.DEATH_EVENT.value_counts().values,labels=['<b>Urban<b>', '<b>Rural<b>'],
           hole=0.3,pull=[0,0.08,0.08,0.08,0.2],marker_colors=['#dcb5ff', '#a5bdfd'],textposition='inside'),
    row=3, col=2
) 

 
    
fig.update_layout(
    height=1200,
    showlegend=True,
    title_text="<b>Distribution of Categorical Varibles<b>",
)

fig.show()



def cat_count(ddf, col):
    ddf = ddf.groupby([col, "DEATH_EVENT"])["DEATH_EVENT"].count().reset_index(level = 0)
    ddf.columns = [col, "count"]
    ddf = ddf.reset_index()
    return ddf


fig, axes = plt.subplots(1, 3, figsize = (30,10));


for ix, i in enumerate(["high_blood_pressure", "sex", "smoking"]):
    xx = cat_count(Heart_Fail1[[i, "DEATH_EVENT"]], i)
    sns.barplot(xx[i], xx["count"], hue = xx["DEATH_EVENT"], palette = "BuPu", ax = axes[ix]);
    
fig, axes = plt.subplots(1, 2, figsize = (30,10));

for ix, i in enumerate(["anaemia", "diabetes"]):
    xx = cat_count(Heart_Fail1[[i, "DEATH_EVENT"]], i)
    sns.barplot(xx[i], xx["count"], hue = xx["DEATH_EVENT"], palette = "BuPu", ax = axes[ix]);



#Visualizing Numerical Features



Heart_Fail1[numerical_features].iplot(kind='histogram', subplots=True,bins=50, colors=['#27296d','#5e63b6','#a393eb','#f5c7f7'],dimensions =(1200,1000))



for i in numerical_features:
    Heart_Fail1[i].iplot(kind="box", title=i, boxpoints="all", color='#a393eb', dimensions=(600,600))


index = 0
plt.figure(figsize=(20,20))
for feature in numerical_features:
    if feature != "DEATH_EVENT":
        index += 1
        plt.subplot(4, 3, index)
        sns.boxplot(x='DEATH_EVENT', y=feature, data=Heart_Fail1, palette='BuPu')


sns.pairplot(Heart_Fail1, hue="DEATH_EVENT", palette="inferno", corner=True);

#Outlier Detection


def detect_outliers(Heart_Fail,features):
    outlier_indices = []
    
    for c in features:
        Q1 = np.percentile(Heart_Fail[c],25)
        Q3 = np.percentile(Heart_Fail[c],75)
        IQR = Q3 - Q1
        outlier_step = IQR * 1.5
        outlier_list_col = Heart_Fail[(Heart_Fail[c] < Q1 - outlier_step) | (Heart_Fail[c] > Q3 + outlier_step)].index 
        outlier_indices.extend(outlier_list_col) 
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1) 
    
    return multiple_outliers


Heart_Fail1.loc[detect_outliers(Heart_Fail1,["age","creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"])]


Heart_Fail1 = Heart_Fail1.drop(detect_outliers(Heart_Fail1,["age","creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]),axis = 0).reset_index(drop=True)
Heart_Fail1


#Correlation Matrix

corr_matrix = Heart_Fail1.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlaation between features")
plt.show()


#Data Normalization

from sklearn.preprocessing import MinMaxScaler,StandardScaler
mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization

# Normalization
Heart_Fail1['age'] = mms.fit_transform(Heart_Fail1[['age']])
Heart_Fail1['creatinine_phosphokinase'] = mms.fit_transform(Heart_Fail1[['creatinine_phosphokinase']])
Heart_Fail1['ejection_fraction'] = mms.fit_transform(Heart_Fail1[['ejection_fraction']])
Heart_Fail1['serum_creatinine'] = mms.fit_transform(Heart_Fail1[['serum_creatinine']])
Heart_Fail1['time'] = mms.fit_transform(Heart_Fail1[['time']])

# Standardization
Heart_Fail1['platelets'] = ss.fit_transform(Heart_Fail1[['platelets']])
Heart_Fail1['serum_sodium'] = ss.fit_transform(Heart_Fail1[['serum_sodium']])
Heart_Fail1.head()


#Data Spliting

x = Heart_Fail1.iloc[:, [4,7,11]].values
y = Heart_Fail1.iloc[:,-1].values


#train Test Split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =0)

#Classification Model
#Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

mylist = []
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
mylist.append(ac)
print(cm)
print(ac)


#KNN

# Finding the optimum number of neighbors 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
list1 = []
for nb in range(3,10):
    classifier = KNeighborsClassifier(n_neighbors=nb, metric='minkowski')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    list1.append(accuracy_score(y_test,y_pred))
plt.plot(list(range(3,10)), list1)
plt.show()


# Training the K Nearest Neighbor Classifier on the Training set

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
mylist.append(ac)
print(cm)
print(ac)

# Finding the optimum number of max_leaf_nodes

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
list1 = []
for leaves in range(2,10):
    classifier = DecisionTreeClassifier(max_leaf_nodes = leaves, random_state=0, criterion='entropy')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    list1.append(accuracy_score(y_test,y_pred))
#print(mylist)
plt.plot(list(range(2,10)), list1)
plt.show()

# Training the Decision Tree Classifier on the Training set

classifier = DecisionTreeClassifier(max_leaf_nodes = 3, random_state=0, criterion='entropy')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print(cm)
print(ac)
mylist.append(ac)

#Finding the optimum number of n_estimators

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
list1 = []
for estimators in range(10,30):
    classifier = RandomForestClassifier(n_estimators = estimators, random_state=0, criterion='entropy')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    list1.append(accuracy_score(y_test,y_pred))
#print(mylist)
plt.plot(list(range(10,30)), list1)
plt.show()

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 11, criterion='entropy', random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
mylist.append(ac)
print(cm)
print(ac)


#pip install xgboost



from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
list1 = []
for estimators in range(10,30,1):
    classifier = XGBClassifier(n_estimators = estimators, max_depth=12, subsample=0.7)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    list1.append(accuracy_score(y_test,y_pred))
#print(mylist)
plt.plot(list(range(10,30,1)), list1)
plt.show()

from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators = 10, max_depth=12, subsample=0.7)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
mylist.append(ac)
print(cm)
print(ac)

model_list = ['Logistic Regression', 'KNearestNeighbours', 'DecisionTree', 'RandomForest',
               'XGBoost']

plt.rcParams['figure.figsize']=10,4 
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=mylist, palette = "BuPu", saturation =2.0)
plt.xlabel('Classifier Models', fontsize = 20 )
plt.ylabel('% of Accuracy', fontsize = 20)
plt.title('Accuracy of Different Classifier Models', fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()

