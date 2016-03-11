# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 18:10:27 2015

@author: Stephane
"""
import os 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import pandas as pd
import numpy as np
import csv
import random as rnd
import os 
import matplotlib.pyplot

os.getcwd()
os.chdir("C:\\Users\\Stephane\\Desktop\\ITAM\\MachinLearning\\4_Perceptron")

data = pd.read_csv("regLin4.csv")



print(data)
print(type(data))

print(data.describe())


X_train, X_test, Y_train, Y_test = train_test_split(data[["X"]],data["y"], train_size=0.75)

ScaleX = preprocessing.StandardScaler()
ScaleX.fit(X_train)
X_train = pd.DataFrame(data = ScaleX.transform(X_train), columns = ['X'])

plt.scatter(X_train,Y_train)

w0_ini = rnd.random()
w_ini = [rnd.random()]

def salida(w0,w,X):
    suma = w0
    for i in range(len(w)):
        suma = suma + w[i]*X.iloc[i]
    if suma > 0:
        res = 1
    else:
        res = 0
    return res

def entrena(w0,w,X_train,Y_train):
    eta = 0.005
    observaciones = len(X_train)
    for i in range(observaciones): 
        error = Y_train.iloc[i] - salida(w0,w,X_train.iloc[i])
        print("Observacion",i,w0,w,error)
        w0 = w0 + eta*error
        columnasdatos = len(X_train.columns)
        for j in range(columnasdatos):
            w[j] = w[j] + eta*error*X_train.iloc[i,j]
    return w0,w
    
w0_fin, w_fin = entrena(w0_ini,w_ini,X_train,Y_train)

-w0_fin/w_fin[0]

Y_Predicted = []


for l in range(len(X_train)):
    suma = w0_fin + w_fin[0]*X_train.iloc[l,0]
    if suma > 0:
        res = 1
    else:
        res = 0
    Y_Predicted = Y_Predicted + [res] 

plt.scatter(X_train,Y_train)
plt.scatter(X_train, Y_Predicted, color = "red", alpha = 0.05)


#################################################################################

neuron_type = "or"

if neuron_type == "and":
    data = pd.DataFrame(np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]]),columns=["X1","X2","Y"])
elif neuron_type == "or":     
    data = pd.DataFrame(np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]]),columns=["X1","X2","Y"])

dataset = data
for i in range(3000):
   dataset = dataset.append(data)

dataset = dataset.reset_index(drop=True)


w0_ini = rnd.random()
w_ini = [rnd.random(),rnd.random()]

def salida_transferencia_and(w0,w,X):
    suma = w0
    for i in range(len(w)):
        suma = suma + w[i]*X.iloc[i]
    if suma > 0.0:
        res = 1
    else:
        res = 0
    return res
    
X = X_train.iloc[0,0:len(X_train.columns)]
X.iloc[0]
salida_transferencia_and(w0_ini,w_ini,X_train.iloc[0,0:len(X_train.columns)])

def entrena_and(w0,w,X_train,Y_train):
    eta = 0.05
    observaciones = len(X_train)
    for i in range(observaciones): 
        error = Y_train.iloc[i,0] - salida_transferencia_and(w0,w,X_train.iloc[i,0:len(X_train.columns)])
        print("Observacion",i,w0,w,error)
        w0 = w0 + eta*error
        columnasdatos = len(X_train.columns)
        for j in range(columnasdatos):
            w[j] = w[j] + eta*error*X_train.iloc[i,j]
    return w0,w
    
X_train = dataset[["X1","X2"]]  
Y_train = dataset[["Y"]]  

w0_fin, w_fin = entrena_and(w0_ini,w_ini,X_train,Y_train)

Y_Predicted = []
for l in range(len(X_train)):
    suma = w0_fin
    for p in range(len(w_fin)):
        suma = suma + w_fin[p]*X_train.iloc[l,p]
    if suma > 0:
        res = 1
    else:
        res = 0
    Y_Predicted += [res]

plt.scatter(X_train[["X1"]],X_train[["X2"]],color="purple")
aux = np.linspace(-0.2,1.2,10)
plt.plot(aux,-aux*w_fin[0]/w_fin[1]-w0_fin/w_fin[1],color="red")
plt.grid()

