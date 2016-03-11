# -*- coding: utf-8 -*-
"""
Created on Thu Sep 03 19:07:44 2015

@author: Stephane Keil Rios
@CVU: 160559
Tarea 4 - LMS Gradient linear regression
"""
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import random as rnd
import os 
import math
import matplotlib.pyplot

os.getcwd()
os.chdir("C:\\Users\\Stephane\\Desktop\\ITAM\\MachinLearning\\3_LinearRegression")


data = pd.read_csv("regLin.csv")

print(data)
print(type(data))

print(data.describe())

X_train, X_test, Y_train, Y_test = train_test_split(data[["X"]],data["y"], train_size=0.75)


ScaleX = preprocessing.StandardScaler()
ScaleY = preprocessing.StandardScaler()

ScaleX.fit(X_train)
X_train = pd.DataFrame(data = ScaleX.transform(X_train), columns = ['X'])

ScaleY.fit(Y_train)
Y_train = pd.Series(data = ScaleY.transform(Y_train), name = 'y' )

w0_ini = rnd.random()
w_ini = [rnd.random()]


def salida(w0,w,X):
    suma = w0
    for i in range(len(w)):
        suma = suma + w[i]*X.iloc[i]
    return suma

def entrena(w0,w,X_train,Y_train):
    eta = 0.1
    observaciones = len(X_train)
    for i in range(observaciones):
        
        sal = salida(w0,w,X_train.iloc[i])
        error = Y_train.iloc[i] - sal
        print("Observacion",i,w0,w,error)
        w0 = w0 + eta*error
        columnasdatos = len(X_train.columns)
        for j in range(columnasdatos):
            w[j] = w[j] + eta*error*X_train.iloc[i,j]
    return w0,w

w0_fin, w_fin = entrena(w0_ini,w_ini,X_train,Y_train)

Y_Predicted = w0_fin + w_fin[0]*X_train
Y_test

plt.scatter(X_train,Y_train)
plt.plot(X_train, Y_Predicted, color='red', linewidth=3)



X_test = pd.DataFrame(data = ScaleX.transform(X_test), columns = ['X'])
Y_test = pd.Series(data = ScaleY.transform(Y_test), name = 'y' )
Y_Predicted_test = w0_fin + w_fin[0]*X_test


#Visualizar los datos y la estimación de la recta producto de la regresion dinamica para el conjunto de prueba
plt.scatter(X_test,Y_test)
plt.plot(X_test, Y_Predicted_test, color='red', linewidth=3)
#Calculo de los errores de la regresion
errores = Y_Predicted_test.iloc[0:len(Y_Predicted_test),0] - Y_test
#Visualizar como se distribuyen los residuales de la regresion
plt.scatter(X_test,errores)
z = np.polyfit(X_test.iloc[0:len(X_test),0], errores, 1)
p = np.poly1d(z)
plt.plot(X_test.iloc[0:len(X_test),0],p(X_test.iloc[0:len(X_test),0]))
#Calculo del error estandar de la estimacion
math.sqrt(np.mean((Y_Predicted_test.iloc[0:len(Y_Predicted_test),0] - Y_test)**2)/len(Y_Predicted_test))
