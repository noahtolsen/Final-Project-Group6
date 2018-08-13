## Final Merged Code for Project Group 6

#------------------------------Naive Bayes--------------------------------------
# Naive Bayes was performed by Jianing Wang

#import the required data packages
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")


#read in the dataset
d=pd.read_csv('cc_default_data.csv', sep=',', header=0)
print(d.head(5))
d.drop(["ID"], axis=1, inplace=True)
# have a look basic information of the data
print("Dataset No. of Rows: ", d.shape[0])
print("Dataset No. of Columns: ", d.shape[1])

# printing the dataset obseravtions
print("Dataset first few rows:\n ")
print(d.head(5))

#check if there are Null variables or not
print("Sum of NULL values in each column. ")
print(d.isnull().sum())

print("Dataset info:\n ")
print(d.info())
print('summary statistics of the dataset:  ')
print(d.describe(include='all'))

#have a look the continuous variable distribution

def distribution(i):
    plt.hist(d.iloc[:,i])
    plt.title('feature distribution')
    plt.show()
distribution(0)

#all the continuous variables don't belong to normal distribution at all

#half of the features are dicrete variables
#use MultinomialNB
#------------------------------MultinomialNB---------------------------------
#-----------------------------data processing--------------------------------

#switch the continuous variables to discrete variables and non-negtitive datas
for j in range(5, 11):
    d.iloc[:,j] = pd.cut(d.iloc[:, j],11 ,labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    d.iloc[:, j] = d.iloc[:, j].astype('float64')

for i in range(11, 23):
    d.iloc[:, i] = pd.cut(d.iloc[:, i], 3,labels=[0, 1, 2])
    d.iloc[:, i] = d.iloc[:, i].astype('float64')


#start training process
#split the data
x=d.values[:,:-1]
y=d.values[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
clf =MultinomialNB(alpha=1, fit_prior=True, class_prior=None)
# performing training
clf.fit(x_train, y_train)
# predicton on test
y_pred = clf.predict(x_test)


y_pred_score = clf.predict_proba(x_test)

print("\n")

print("Classification Report of Multinomial : ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy of Multinomial : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC of Multinomial : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
print("\n")


#correlation map
data = d.corr()
sns.heatmap(data)
plt.title('Correlation')
plt.show()


#delete high corelated columns and train again
d.drop(['BILL_AMT1','BILL_AMT3','BILL_AMT6'],axis=1, inplace=True)
x=d.values[:,:-1]
y=d.values[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
clf =MultinomialNB(alpha=1, fit_prior=True, class_prior=None)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

y_pred_score = clf.predict_proba(x_test)

print("\n")

print("Classification Report of Multinomial_2 : ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy of Multinomial_2 : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC of Multinomial_2 : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
print("\n")

#show confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = class_names = ['0','1']

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
cm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
cm.yaxis.set_ticklabels(cm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
cm.xaxis.set_ticklabels(cm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.title('Confusion matrix of Multinomial')
plt.show()
#even after droping the high corelated columns, the accuracy is not high evough.


#let's try use GaussianNB()
#------------------------------GaussianNB--------------------------------------
#-----------------------------data processing----------------------------------
#normalizing the data
d_gaussian=pd.read_csv('cc_default_data.csv', sep=',', header=0)
d_gaussian.drop(["ID"], axis=1, inplace=True)
predata=d_gaussian.values[:,(11,12,13,14,15,16,17,18,19,20,21,22)]

scaledata=preprocessing.scale(predata)
scaledata.shape
x1=d_gaussian.values[:,:11]
x=np.c_[x1,scaledata]


#start training process
#split the data
y=d_gaussian.values[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
clf =GaussianNB()
# performing training
clf.fit(x_train, y_train)
# predicton on test
y_pred = clf.predict(x_test)

y_pred_score = clf.predict_proba(x_test)

print("\n")

print("Classification Report of Gaussian: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy of Gaussian : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC of Gaussian: ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
print("\n")

#show confusion matrix of Gaussian
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = class_names = ['0','1']

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
cm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
cm.yaxis.set_ticklabels(cm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
cm.xaxis.set_ticklabels(cm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.title('Confusion matrix of Gaussian')
plt.tight_layout()
plt.show()





# ---------------------------- SVM --------------------------------------
# The Support Vector Machine was performed by Jia Chen
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
from pandas.core.frame import DataFrame
warnings.filterwarnings("ignore")


# %%-----------------------------------------------------------------------
# importing Data
# read data as pandas dataframe
data = pd.read_csv("cc_default_data_SVM.csv")

# define column names
data.columns = ['ID', 'LimitBalance', 'Sex', 'Education', 'MaritalStatus', 'Age', 'Repayment_Sept',
                'Repayment_Aug', 'Repayment_July', 'Repayment_June', 'Repayment_May', 'Repayment_Apr',
                'BillAmt_Sept', 'BillAmt_Aug', 'BillAmt_July', 'BillAmt_June','BillAmtMay', 'BillAmt_Apr',
                'PaymentAmt_Sept', 'PaymentAmt_Aug', 'PaymentAmt_July', 'PaymentAmt_June', 'PaymentAmt_May',
                'PaymentAmt_Apr', 'default payment next month']

# %%-----------------------------------------------------------------------
# Data pre-processing


# drop unnecessary rows and columns
data.drop([0], inplace=True)
data.drop(['ID'], axis=1, inplace=True)
# look at the first few rows
print(data.head())
# print(data.columns.tolist())


# replace missing characters as NaN
data = data.replace('?', np.NaN, inplace=False)
# check the structure of data
data.info()
# check the null values in each column
print(data.isnull().sum())
# check the summary of the data
data.describe(include='all')


# normalize continuous columns such as LimitBalance, BillAmount and PaymentAmount
X_Part1 = data.values[:, :1]
X_Part2 = data.values[:, 11:23]
X_Part5 = data.values[:, 4:5]
min_max_data = preprocessing.MinMaxScaler()

# transfer them into dataframe to merge with other parts
X_Part1_minmax = DataFrame(min_max_data.fit_transform(X_Part1))
X_Part2_minmax = DataFrame(min_max_data.fit_transform(X_Part2))
X_Part5_minmax = DataFrame(min_max_data.fit_transform(X_Part5))

# adjust index to be consistent with get_dummies dataframe below
X_Part1_minmax.index = range(1, len(X_Part1_minmax)+1)
X_Part2_minmax.index = range(1, len(X_Part2_minmax)+1)
X_Part5_minmax.index = range(1, len(X_Part5_minmax)+1)

# some features keep unchanging
# X_Part4 = DataFrame(data.values[:, 5:11])

# X_Part4.index = range(1, len(X_Part4)+1)

# %%-----------------------------------------------------------------------
# One Hot Encoding the variables

# encoding categorical features such as gender, education, and marriage status using get dummies
X_Part3 = pd.get_dummies(data.iloc[:, 1:4])
X_Part4 = pd.get_dummies(data.iloc[:, 5:11])

# merge all the parts above
X_Entire = pd.concat([X_Part1_minmax, X_Part2_minmax, X_Part3, X_Part4, X_Part5_minmax], axis=1)

X_data = X_Entire.values
X = X_data[:, :]

# encoding the class with sklearn's LabelEncoder
Y_data = data.values[:, 23]
class_le = LabelEncoder()

# fit and transform the class
y = class_le.fit_transform(Y_data)

# %%-----------------------------------------------------------------------

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# %%----------------------------------------------------------------------- Train
# perform training

# creating the classifier object
clf = SVC(kernel="linear")

# performing training
clf.fit(X_train, y_train)
# %%----------------------------------------------------------------------- Predict

# make predictions

# predict on test

y_predict = clf.predict(X_test)
print(sum(y_predict))

# ----------------------------------------------------------------------- Accuracy

# calculate metrics

print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_predict))
print("\n")

# -----------------------------------------------------------------------
# feature importance, confusion matrix, ROC area mainly used professor's code
# function to display feature importance of the classifier

# display top 20 features (top 10 max positive and negative coefficient values)


def coef_values(coef, names):
    imp = coef
    print(imp)
    imp,names = zip(*sorted(zip(imp.ravel(),names)))

    imp_pos_10 = imp[-10:]
    names_pos_10 = names[-10:]
    imp_neg_10 = imp[:10]
    names_neg_10 = names[:10]

    imp_top_20 = imp_neg_10+imp_pos_10
    names_top_20 = names_neg_10+names_pos_10

    plt.barh(range(len(names_top_20)), imp_top_20, align='center')
    plt.yticks(range(len(names_top_20)), names_top_20)
    plt.show()


# get the column names
features_names = X_Entire.columns.tolist()
# call the function
coef_values(clf.coef_, features_names)

# -----------------------------------------------------------------------

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)
# class_names = data['default payment next month'].unique()
class_names = class_names = ['0', '1']


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(5, 5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------
# Plot ROC Area Under Curve

y_predict_probability = clf.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test,  y_predict_probability)
auc = roc_auc_score(y_test, y_predict_probability)

# print(fpr)
# print(tpr)
# print(auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# ---------------------------- Neural Network --------------------------------------
# The Neural Network was performed by Noah Olsen


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('cc_default_data.csv', sep=',', header=0)
X = data.values[:, :-1]
Y = data.values[:, -1]


# printing the summary statistics of the dataset
print(data.describe(include='all'))

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1996)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)


predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# ---------------------------- Random Forest --------------------------------------
# The Random Forest was performed by Noah Olsen


# Importing the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


#%%-----------------------------------------------------------------------
# import Dataset
# read data as panda dataframe
data = pd.read_csv('cc_default_data.csv', sep=',', header=0)

# printing the dataswet rows and columns
print("Dataset No. of Rows: ", data.shape[0])
print("Dataset No. of Columns: ", data.shape[1])

# printing the dataset obseravtions
print("Dataset first few rows:\n ")
print(data.head(2))

# printing the struture of the dataset
print("Dataset info:\n ")
print(data.info())

# printing the summary statistics of the dataset
print(data.describe(include='all'))
#%%-----------------------------------------------------------------------
#clean the dataset
print("Sum of NULL values in each column. ")
print(data.isnull().sum())

# drop unnnecessary columns
data.drop(["ID"], axis=1, inplace=True)


#%%-----------------------------------------------------------------------
#split the dataset
# separate the predictor and target variable
X = data.values[:, :-1]
Y = data.values[:, -1]

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1996)
#%%-----------------------------------------------------------------------
#perform training with random forest with all columns
# specify random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# perform training
clf.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
#plot feature importances
# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, data.iloc[:, :-1].columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
#select features to perform training with random forest with k columns
# select the training dataset on k-features
newX_train = X_train[:, clf.feature_importances_.argsort()[::-1][:15]]

# select the testing dataset on k-features
newX_test = X_test[:, clf.feature_importances_.argsort()[::-1][:15]]

#%%-----------------------------------------------------------------------
#perform training with random forest with k columns
# specify random forest classifier
clf_k_features = RandomForestClassifier(n_estimators=1000)

# train the model
clf_k_features.fit(newX_train, y_train)

#%%-----------------------------------------------------------------------
#make predictions

# predicton on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

# prediction on test using k features
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)


# %%-----------------------------------------------------------------------
# calculate metrics gini model

print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)

# calculate metrics entropy model
print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_k_features))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_k_features_score[:,1]) * 100)

# %%-----------------------------------------------------------------------
# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = class_names = ['0','1']


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()


# %%-----------------------------------------------------------------------

# confusion matrix for entropy model

conf_matrix = confusion_matrix(y_test, y_pred_k_features)
class_names = class_names = ['0','1']


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()
