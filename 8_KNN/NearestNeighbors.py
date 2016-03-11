# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 20:23:19 2015

@author: Stephane
"""

import pandas as pd
import numpy as np
import csv
import random
import math
import matplotlib.pyplot as plt
import os
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm


os.getcwd()
os.chdir("C:\\Users\\Stephane\\Desktop\\ITAM\\MachinLearning\\8_KNN")
os.listdir(".")

lower=0
upper=20

n=1000
X1 = map(random.uniform,[lower]*n,[upper]*n)
X2 = map(random.uniform,[lower]*n,[upper]*n)

circulos = 5
radio=2
centrox1 =  map(random.uniform,[lower]*circulos,[upper]*circulos)
centrox2 =  map(random.uniform,[lower]*circulos,[upper]*circulos)


def Clasifica(X1,X2,centrox1,centrox2,radio):
    Y = []
    for i in range(len(X1)):
        checa=0
        for j in range(len(centrox1)):
            value = math.sqrt((X1[i]-centrox1[j])**2+(X2[i]-centrox2[j])**2)
            if value <= radio:
                checa = 1
        Y.append(checa)
    data = pd.DataFrame(zip(X1,X2,Y), columns = ["X1","X2","Y"])
    return data
            
train = Clasifica(X1,X2,centrox1,centrox2,radio)


train.groupby('Y').count()

plt.scatter(train['X1'], train['X2'], c=train['Y'], linewidths=0)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train[["X1","X2"]],train["Y"]) 


clf = svm.SVC(kernel='rbf', gamma=20, C=20)
clf.fit(train[["X1", "X2"]], train["Y"])

xx, yy = np.meshgrid(np.linspace(lower,upper,100),
                     np.linspace(lower,upper,100))
                     
         
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
          origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linetypes='--')
plt.scatter(train['X1'], train['X2'], s=30, c=train['Y'], cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([lower,upper,lower,upper])
plt.show()


'''Conjunto de pruebas'''



ntest=1000
X1test = map(random.uniform,[lower]*ntest,[upper]*ntest)
X2test = map(random.uniform,[lower]*ntest,[upper]*ntest)


'''Nearest Neighbors'''
test = Clasifica(X1test,X2test,centrox1,centrox2,radio)

plt.scatter(test['X1'], test['X2'], c=test['Y'], linewidths=0)
plt.scatter(test['X1'], test['X2'], c=Y_predict, linewidths=0)

Y_predict = neigh.predict(test[["X1","X2"]])
test["Y_predict"] = Y_predict 

matriz_confusion = [] 
for i in range(len(test.index)):
    if test["Y"][i] == 0:
        if test["Y_predict"][i] == 0:
            matriz_confusion.append("TN")
        else:
            matriz_confusion.append("FP")
    else:
        if test["Y_predict"][i] == 0:
            matriz_confusion.append("FN")
        else:
            matriz_confusion.append("TP")

test.loc[:,"matriz_confusion"] = matriz_confusion
conf = test.groupby("matriz_confusion").count()["Y_predict"]
sum(conf[0:2].astype('float'))/sum(conf[0:4].astype('float'))
###0.033000000000000002 - 10 neighbors
###0.029000000000000001 - 5 neighbors
###0.023 - 1 neighbor

'''SVM test'''


test_SVM = test.loc[:,['X1','X2','Y']]

Y_predict_svm = clf.decision_function(zip(test_SVM.iloc[:,0],test_SVM.iloc[:,1]))
plt.scatter(test_SVM['X1'], test_SVM['X2'], s=30, c=test_SVM['Y'], cmap=plt.cm.Paired)
plt.scatter(test_SVM['X1'], test_SVM['X2'], s=30, c=Y_predict_svm, cmap=plt.cm.Paired)
test_SVM["Y_predict"] = Y_predict_svm
test_SVM.groupby("Y").mean()

fpr, tpr, thresholds = roc_curve(test_SVM.iloc[:,2], Y_predict_svm)

dist=  map(math.sqrt,(1-tpr)**2+(fpr**2))
ind = dist.index(min(dist))
thresholds[ind]


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr, tpr, lw=1, label='Curva ROC')
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.title("Curva Roc")
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Suerte')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
for xy in range(0,1000,75):  
    la = round(thresholds[xy],2)
    fi = (fpr[xy],tpr[xy])                                             # <--
    ax.annotate('(%s)' % la, xy=fi, textcoords='offset points', size='small')
plt.show()

umbral = thresholds[ind]
Y_predict_coded = []
for i in Y_predict_svm:
    if i > umbral:
        Y_predict_coded.append(1)
    else:
        Y_predict_coded.append(0)

test_SVM.loc[:,"Y_predict_coded"] = Y_predict_coded
matriz_confusion = [] 
for i in range(len(test_SVM.index)):
    if test_SVM["Y"][i] == 0:
        if test_SVM["Y_predict_coded"][i] == 0:
            matriz_confusion.append("TN")
        else:
            matriz_confusion.append("FP")
    else:
        if test_SVM["Y_predict_coded"][i] == 0:
            matriz_confusion.append("FN")
        else:
            matriz_confusion.append("TP")

test_SVM.loc[:,"matriz_confusion"] = matriz_confusion
conf_SVM = test_SVM.groupby("matriz_confusion").count()["Y_predict_coded"]
plt.scatter(test[["X1"]],test[["X2"]], color=['green' if i=="TP" else 'blue' if i == "FP" else 'yellow' if i == "FN" else 'red' for i in matriz_confusion], linewidths=0)
sum(conf_SVM[0:2].astype('float'))/sum(conf_SVM[0:4].astype('float'))
### 0.023 - gamma =1 C=1
### 0.029 - gamma= 1 C=20
### 0.025 - gamma=10 C=1
### 0.024 - gamma=10 C=20
### 0.024 - gamma=20 C=1
### 0.024 - gamma =20 C=20