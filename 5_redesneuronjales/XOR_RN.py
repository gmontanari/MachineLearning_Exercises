# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 00:21:32 2015

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
from sklearn.metrics import roc_curve, auc

os.getcwd()
os.chdir("C:\\Users\\Stephane\\Desktop\\ITAM\\MachinLearning\\5_redesneuronjales")
os.listdir(".")

#data = pd.DataFrame(np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]]),columns=["X1","X2","Y"])
#
#dataset = data
#for i in range(100):
#   dataset = dataset.append(data)
#
#dataset = dataset.reset_index(drop=True)


'''Creo datos entrenamiento'''
ntrain = 1000
X1 = map(random.uniform,[0]*ntrain,[1]*ntrain)
X2 = map(random.uniform,[0]*ntrain,[1]*ntrain)
Y = []
for i in range(ntrain):
    if (X1[i] > 0.5 and X2[i]>0.5) or (X1[i] <= 0.5 and X2[i]<=0.5):
        Y.append(0)
    else:
        Y.append(1)
dataset = pd.DataFrame(zip(X1,X2,Y), columns = ["X1","X2","Y"])


X_train = dataset[["X1","X2"]]  
Y_train = dataset[["Y"]]  


'''Arquitectura de la red neuronal'''
net = buildNetwork(2, 4, 1)#### Neuronas en capa entrada, cuantas neuronas en capa intermedia, capa final
ds = SupervisedDataSet(2, 1)
ds.setField('input', X_train)
ds.setField('target', Y_train)
plt.scatter(ds['input'][:,0], ds['input'][:,1], c=ds['target'], linewidths=0)

'''Entreno mi red neuronal'''
trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(500)

'''Creo Datos de prueba'''
ntest = 1000
X1test = map(random.uniform,[0]*ntest,[1]*ntest)
X2test = map(random.uniform,[0]*ntest,[1]*ntest)
Ytest = []
for i in range(ntest):
    if (X1test[i] > 0.5 and X2test[i]>0.5) or (X1test[i] <= 0.5 and X2test[i]<=0.5):
        Ytest.append(0)
    else:
        Ytest.append(1)
test = pd.DataFrame(zip(X1test,X2test,Ytest), columns = ["X1","X2","Y"])

'''Grafico mis datos de prueba'''
plt.scatter(test['X1'], test['X2'], c=test['Y'], linewidths=0)
    

X_test = test[["X1","X2"]]
Y_test = test[["Y"]]

Y_predict = []
for i in range(ntest):
    Y_predict.append(net.activate(X_test.iloc[i,:])[0])


plt.scatter(test['X1'], test['X2'], c=Y_predict, linewidths=0)

test["Y_predict"] = Y_predict

test.groupby("Y").mean()


fpr, tpr, thresholds = roc_curve(test.iloc[:,2], Y_predict)
auc = auc(fpr,tpr)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr, tpr, lw=1, label='Curva ROC')
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.title("Curva Roc")
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Suerte')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
for xy in range(0,999,100):  
    la = round(thresholds[xy],2)
    fi = (fpr[xy],tpr[xy])                                             # <--
    ax.annotate('(%s)' % la, xy=fi, textcoords='offset points', size='small')
plt.show()


'''Metdodo para encontrar el umbral que minimiza la distancia al punto 0,1'''
dist=  map(math.sqrt,(1-tpr)**2+(fpr**2))
ind = dist.index(min(dist))
thresholds[ind]

umbral = 0.4538432749322498
Y_predict_coded = []
for i in Y_predict:
    if i > umbral:
        Y_predict_coded.append(1)
    else:
        Y_predict_coded.append(0)

test.loc[:,"Y_predict_coded"] = Y_predict_coded
matriz_confusion = [] 
for i in range(len(test.index)):
    if test["Y"][i] == 0:
        if test["Y_predict_coded"][i] == 0:
            matriz_confusion.append("TN")
        else:
            matriz_confusion.append("FP")
    else:
        if test["Y_predict_coded"][i] == 0:
            matriz_confusion.append("FN")
        else:
            matriz_confusion.append("TP")

test.loc[:,"matriz_confusion"] = matriz_confusion
test.groupby("matriz_confusion").count()["Y_predict_coded"]
plt.scatter(test[["X1"]],test[["X2"]], color=['green' if i=="TP" else 'blue' if i == "FP" else 'yellow' if i == "FN" else 'red' for i in matriz_confusion], linewidths=0)

conf = confusion_matrix(test.iloc[:,2], test.iloc[:,4])
cm_normalized = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]

plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["XOR False","XOR True"], rotation=45)
plt.yticks(tick_marks, ["XOR False","XOR True"])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

