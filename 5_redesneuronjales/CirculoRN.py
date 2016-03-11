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
from sklearn.metrics import roc_curve

os.getcwd()
os.chdir("C:\\Users\\Stephane\\Desktop\\ITAM\\MachinLearning\\5_redesneuronjales")
os.listdir(".")


"""
Este metodo genera coordenadas aleatorias alrededor del circulo de radio r y de centro (centerx1,centerx2) con un extra de 0.2 en todas las direcciones
n= numero de puntos a generar
r=radio del circulo
centerx1 = coordenada en absisas del centro del circulo
centerx2 = coordenada en ordenadas del centro del circulo

Devuelve 1 si pertenece al circulo de radio r y centro (centerx1,centerx2)
0 si no pertence 

"""
def CreaDatosCirculoNet(n,r=1,centerx1=0,centerx2=0):
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


data1 = CreaDatosCirculoNet(300,1,0,0)
plt.scatter(data1['X1'], data1['X2'], c=data1['Y'], linewidths=0)
""" Preparación de los parametros de mi red neuronal
"""


net1 = buildNetwork(2, 50, 1)#### Neuronas en capa entrada, cuantas neuronas en capa intermedia, capa final
ds1 = SupervisedDataSet(2, 1)

ds1.setField('input', data1[["X1","X2"]])
ds1.setField('target', data1[["Y"]])

plt.scatter(ds1['input'][:,0], ds1['input'][:,1], c=ds1['target'], linewidths=0)

"""
Entreno mi red neuronal
"""

trainer1 = BackpropTrainer(net1, ds1)

trainer1.trainEpochs(1000)
"""
Genero un conjunto de datos de prueba

"""
test1 = CreaDatosCirculoNet(10000,1,0,0)
X_test1 = test1[["X1","X2"]]
Y_test1 = test1[["Y"]]


"""Uso la red neuronal entrenada sobre mi conjunto de pruba
"""


Y_predict1 = []
for i in range(len(X_test1.index)):
    Y_predict1.append(net1.activate(X_test1.iloc[i,:])[0])

"""
Grafico mi conjunto de prueba y sus valores predecidos
"""
plt.scatter(test1['X1'], test1['X2'], c=Y_predict1, linewidths=0)

Y_test1["Y_predict"] = Y_predict1
Y_test1.groupby('Y').mean()### Rule of thumb para mi umbral inicial


"""Clasifico mis observaciones predecidas confirme a un valor de umbral
que selecciono de mis curvas ROC
"""

umbral1 = 0.54731829841616242### Este umbral se selecciona de mis curvas ROC
Y_predict_coded1 = []
for i in Y_predict1:
    if i > umbral1:
        Y_predict_coded1.append(1)
    else:
        Y_predict_coded1.append(0)
Y_test1.loc[:,"Y_predict_coded"] = Y_predict_coded1
""" Calculo los valores de la matriz de confusion de manera manual
"""

matriz_confusion1 = [] 
for i in range(len(Y_test1.index)):
    if Y_test1["Y"][i] == 0:
        if Y_test1["Y_predict_coded"][i] == 0:
            matriz_confusion1.append("TN")
        else:
            matriz_confusion1.append("FP")
    else:
        if Y_test1["Y_predict_coded"][i] == 0:
            matriz_confusion1.append("FN")
        else:
            matriz_confusion1.append("TP")


""" 

"""
Y_test1.loc[:,"matriz_confusion"] = matriz_confusion1
Y_test1.groupby("matriz_confusion").count()["Y_predict_coded"]
"""Grafico los valores de prueba alrededor del circulo con codigo de colores
"""


plt.scatter(X_test1[["X1"]],X_test1[["X2"]], color=['green' if i=="TP" else 'blue' if i == "FP" else 'yellow' if i == "FN" else 'red' for i in matriz_confusion1], linewidths=0)

"""Calculo mi matriz de confusion con el metodo de scikit Renglon1 TN, FP//Renglon 2 FN, TP
"""

conf1 = confusion_matrix(Y_test1.iloc[:,0], Y_test1.iloc[:,2])
cm_normalized1 = conf1.astype('float') / conf1.sum(axis=1)[:, np.newaxis]

""" Grafico la matriz de confusion
"""


plt.imshow(cm_normalized1, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Not in Circle","In Circle"], rotation=45)
plt.yticks(tick_marks, ["Not in Circle","In Circle"])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

"""Genero la curva ROC para determinar el umbral que permite tener el mayor numero 
de verdaderos positivos y el menor numero de falsos positivos
"""


fpr1, tpr1, thresholds1 = roc_curve(Y_test1.iloc[:,0], Y_predict1)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr1, tpr1, lw=1, label='Curva ROC')
plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.title("Curva Roc")
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Suerte')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
for xy in range(0,9999,1000):  
    la = round(thresholds1[xy],2)
    fi = (fpr1[xy],tpr1[xy])                                             # <--
    ax.annotate('(%s)' % la, xy=fi, textcoords='offset points', size='small')
plt.show()

""" Realizo la curva ROC con zoom al area de interes para determinar el mejor umbral
"""
    
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(fpr1, tpr1, lw=1, label='Curva ROC')
plt.axis([-0.05, 0.3, 0.75, 1.05])
plt.title("Curva Roc")
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Suerte')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
for xy in range(4000,6500,200):  
    la = round(thresholds1[xy],2)
    fi = (fpr1[xy],tpr1[xy])                                             # <--
    ax.annotate('(%s)' % la, xy=fi, textcoords='offset points', size='small')
plt.show()    

'''Metdodo para encontrar el umbral que minimiza la distancia al punto 0,1'''
dist1=  map(math.sqrt,(1-tpr1)**2+(fpr1**2))
ind1 = dist1.index(min(dist1))
thresholds1[ind1]

'''Obtener indices en un numpy array'''
dir(fpr1[(fpr1 > 0) & (fpr1 < 0.05)]) 
thresholds1[np.where((fpr1 > 0) & (fpr1 < 0.05))]

'''Obtener indices en un pandas data frame'''
test = pd.DataFrame(thresholds1)
test2 = pd.DataFrame(fpr1[np.asarray(test[(test[0] > 0.33) & (test[0] < 0.65)].index)], index =np.asarray(test[(test[0] > 0.33) & (test[0] < 0.65)].index))
test2[(test2[0] < 0.05)]
test.iloc[5196]