# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:41:31 2015

@author: stephanekeil
"""


import pandas
import pandas as pd
import numpy as np
import csv
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy
import os
import random
import math
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn import svm
from matplotlib import style
style.use("ggplot")

os.getcwd()
os.chdir("C:\\Users\\Stephane\\Desktop\\ITAM\\MachinLearning\\7_SVM")
os.listdir(".")


m1=pandas.read_csv("andSVM.csv")
#m1

X1=m1.iloc[:,0]
X2=m1.iloc[:,1]
y=m1.iloc[:,2]


plt.scatter(X1, X2, c=y, linewidths=0)



clf = svm.SVC(kernel='linear', C = 20.0)
clf.fit(m1[["X1", "X2"]],m1["y"])

w0=clf.intercept_[0]
w1=clf.coef_[0,0]
w2=clf.coef_[0,1]

print(w0,w1,w2)

'''
Y = w0 +w1*x1 +w2*x2

Igualo Y a 0
x2 = -w0/w2 -w1/w0*x1

Sabemos que los margenes que pasan por soportes vectoriales tienen la misma pendiente
Yo se que SV1 es un punto de coord (x1s,x2s) que es un soporte vectorial

X2s = X1s*m + intercepto
intercepto = X2s-X1s*m
Y= X*m + intercepto
Y= X*m + (X2s-X1s*m)

'''

x_ax = np.linspace(min(X1)-0.5,max(X1)+0.5,50)
m = -w1/w2
x2 = -w0/w2+m*x_ax


supportlow = clf.support_vectors_[0]
supporthigh = clf.support_vectors_[-1]

down = x_ax*m+(supportlow[1]-supportlow[0]*m)
high = x_ax*m+(supporthigh[1]-supporthigh[0]*m)


plt.scatter(X1, X2, c=y)
plt.plot(x_ax,down,linewidth=1, linestyle='dashed',color='blue')
plt.plot(x_ax,high,linewidth=1, linestyle='dashed',color='blue')
plt.plot(x_ax,x2,linewidth=3,color='red')
plt.show()

help(plt.plot)
dir(clf)
help(clf.coef_) # es el help de python

'''
SVM para un circulo
'''

def CreaDatosCirculo(n,r=1,centerx1=0,centerx2=0):
    X1 = map(random.uniform,[centerx1-r-0.2]*n,[centerx1+r+0.2]*n)
    X2 = map(random.uniform,[centerx2-r-0.2]*n,[centerx2+r+0.2]*n)    
    Y = []
    for i in range(len(X1)):
        valor = math.sqrt(X1[i]**2+X2[i]**2)
        if valor <= 1:
            Y.append(1)
        else:
            Y.append(0)
    data = pd.DataFrame(zip(X1,X2,Y), columns = ["X1","X2","Y"])
    return data
    
"""
Genero mi conjunto de datos de entrenamiento

"""


data = CreaDatosCirculo(600,1,0,0)
plt.scatter(data['X1'], data['X2'], c=data['Y'], linewidths=0)

'''Kernel linear'''
clf = svm.SVC(kernel='linear', C=50)
clf.fit(data[["X1", "X2"]], data["Y"])

xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 100),
                     np.linspace(-1.2, 1.2, 100))
                     
         
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
          origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linetypes='--')
plt.scatter(data['X1'], data['X2'], s=30, c=data['Y'], cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.show()


'''Kernel Polinomial'''
clf = svm.SVC(kernel='poly',  gamma=20)
clf.fit(data[["X1", "X2"]], data["Y"])

xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 100),
                     np.linspace(-1.2, 1.2, 100))
                     
         
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
          origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linetypes='--')
plt.scatter(data['X1'], data['X2'], s=30, c=data['Y'], cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.show()


'''Kernel RBF'''
clf = svm.SVC(kernel='rbf', gamma=1)
clf.fit(data[["X1", "X2"]], data["Y"])

xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 100),
                     np.linspace(-1.2, 1.2, 100))
                     
         
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
          origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linetypes='--')
plt.scatter(data['X1'], data['X2'], s=30, c=data['Y'], cmap=plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.show()


'''Set de prueba'''

test = CreaDatosCirculo(600,1,0,0)
Y_predict = clf.decision_function(zip(test.iloc[:,0],test.iloc[:,1]))
plt.scatter(test['X1'], test['X2'], s=30, c=test['Y'], cmap=plt.cm.Paired)
plt.scatter(test['X1'], test['X2'], s=30, c=Y_predict, cmap=plt.cm.Paired)


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
for xy in range(0,600,100):  
    la = round(thresholds[xy],2)
    fi = (fpr[xy],tpr[xy])                                             # <--
    ax.annotate('(%s)' % la, xy=fi, textcoords='offset points', size='small')
plt.show()


'''Zoom a la  curva ROC'''
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr, tpr, lw=1, label='Curva ROC')
plt.axis([-0.05, 0.1, 0.90, 1.05])
plt.title("Curva Roc")
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Suerte')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
for xy in range(270,340,3):  
    la = round(thresholds[xy],2)
    fi = (fpr[xy],tpr[xy])                                             # <--
    ax.annotate('(%s)' % la, xy=fi, textcoords='offset points', size='small')
plt.show()   

umbral = 0.0
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
plt.xticks(tick_marks, ["Not in Circle","In Circle"], rotation=45)
plt.yticks(tick_marks, ["Not in Circle","In Circle"])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

