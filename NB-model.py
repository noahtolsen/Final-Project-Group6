#------------------------------Naive Bayes--------------------------------------

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
class_names = d['default payment next month'].unique()

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
class_names = d['default payment next month'].unique()

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
