# -*- coding: utf-8 -*-
"""
Created on Thu Sep 03 19:07:44 2015

@author: Stephane
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
"""
En esta seccion se establece el working dir donde residen los datos
"""
matplotlib.style.use('ggplot')
os.getcwd()
os.chdir("C:\\Users\\Stephane\\Desktop\\ITAM\\MachinLearning\\3_LinearRegression")

###Leo los datos 
data = pd.read_csv("regLinPoli.csv")
###Creo una tabla donde se guardaran los RMS errors de los modelos de gradiente tradicional y con regularización de Ridge
errores_modelos = pd.DataFrame(index=[1], columns=["gradiente","regularizado"])


"""
Esta sección busca tener una idea del comportamiento de los datos
Es un breve analisis exploratorio


Saltarla por favor si se quire el metodo regularizado
"""
###Ver los datos y un summary
print(data)
print(type(data))
print(data.describe())



###Grafico sus funciones de densidad
for f in range(data.shape[1]):
   plt.figure(f+1)
   data[[f]].plot(kind = "density")
   plt.show()
   
###Grafico las suma acumulativa
for f in range(data.shape[1]):
   plt.figure(f+1)
   data[[f]].cumsum().plot()
   plt.show()

###Grafico las sumas cumulativas de un jalon
plt.figure(); data.cumsum().plot();

data[[0]].plot(kind = "density")
   
###Histogramas de cada variable
for f in range(data.shape[1]):
   plt.figure(f+1)
   data[[f]].hist(bins=200)
   plt.show()

### Histograma de todas las variables
plt.figure(); data.hist(bins=100)
   
### Boxplot & Whisker de cada variable   
for f in range(data.shape[1]):
   plt.figure(f+1)
   data[[f]].boxplot()
   plt.show()
   
###Boxplot de todas las variables
plt.figure(); data.boxplot()

### Observo el comportamiento de las variables indpendientes con la dependiente, añado la recta de una regresión linear
for f in range(data.shape[1]-1):
    plt.figure(f+1)
    plt.scatter(data[[f]],data[[4]])
    z = np.polyfit(data.iloc[0:data.shape[0],f], data.iloc[0:data.shape[0],4], 1)
    p = np.poly1d(z)
    plt.plot(data[[f]],p(data[[f]]),color="red", linewidth=3)
    plt.show()

"""
Aqui empieza el preprocesamiento de los datos
"""
#### Separo mi conjunto de datos en entrenamiento 75% y prueba 25% 
X_train, X_test, Y_train, Y_test = train_test_split(data[["X","X2","X3","X4"]],data["y"], train_size=0.75)

#### Preparo los metodos de estandarización de los datos y uso una normalización clasica a N(0,1)
ScaleX = preprocessing.StandardScaler()
ScaleY = preprocessing.StandardScaler()

#### Unicamente escalo el test de entrenamiento
ScaleX.fit(X_train)
X_train = pd.DataFrame(data = ScaleX.transform(X_train), columns = ["X","X2","X3","X4"])

ScaleY.fit(Y_train)
Y_train = pd.Series(data = ScaleY.transform(Y_train), name = 'y' )

"""
Inicia el modelo de gradiente descendente sin el uso de regularización de ridge
"""

###w0 es el intercepto y w es un vector de coefficientes de regresion lineal
###Los genero totalmente aleatorios uniformes entre 0 y 1
###Por cierto que Python usa el Mersenne Twister que hablamos en compustats
w0_ini = rnd.random()
w_ini = [rnd.random() for i in xrange(X_train.shape[1])]


def salida(w0,w,X):
    """
    Recibe una w0 y un vector de coefficientes lineales
    Recibe un vector de valores X de una observación del conjunto de datos
    
    Calcula una y predecida que llamamos la salida del modelo
    """
    suma = w0
    for i in range(len(w)):
        suma = suma + w[i]*X.iloc[i]
    return suma

def entrena(w0,w,X_train,Y_train):
    """
    Metodo de entrenamiento de gradiente descendente sin regularización de ridge
    Recibe una w0 y un vector de w iniciales 
    Recibe una matriz de valores que contienen las variables independientes por columna
    Recibe un vector con las observaciones de la variable dependiente
    
    """
    eta = 0.1#Peso que se le da al error de predicción debe ser menor pequeño respecto a las w para que el metodo converga
    observaciones = len(X_train)
    #### Por cada nuevo valor que me va llegando calculo mis nuevas w's considerando el error de predicción usando las w's actuales
    for i in range(observaciones):
        
        sal = salida(w0,w,X_train.iloc[i,0:len(X_train.columns)])
        #### Calculo el error de predicción de y gorro
        error = Y_train.iloc[i] - sal
        ###Actualizo mi w0        
        w0 = w0 + eta*error
        columnasdatos = len(X_train.columns)
        ### Actualizo mis w's
        for j in range(columnasdatos):
            w[j] = w[j] + eta*error*X_train.iloc[i,j]
    ### Regreso las w0 y w's wue calcule al final
    return w0,w

### Mando llamar el metodo de gradiante descendente para entrenar mi modelo
w0_fin, w_fin = entrena(w0_ini,w_ini,X_train,Y_train)
### Calculo mis predicciones de y con las w's obentidas del modelo
Y_Predicted = w0_fin + np.dot(X_train,w_fin)
###Calculo el error cuadratico medio
np.mean((Y_train - Y_Predicted )**2)


### Observemos si la prediccion se parece a los datos
for f in range(X_train.shape[1]):
    plt.figure(f+1)
    plt.scatter(X_train.iloc[0:X_train.shape[0],f],Y_train)
    plt.scatter(X_train.iloc[0:X_train.shape[0],f],Y_Predicted, color='red')
    plt.axis([-2, 2.5, -2, 4])
    plt.show()


#### Calculo el RMSerror para este metodo
errores_modelos.iloc[0,0] = math.sqrt(np.mean((Y_train - Y_Predicted )**2))
"""
A partir de aqui empieza la sección dónde se utiliza el metodo de regularización de ridge 

"""
def entrena_regularizado(w0,w,X_train,Y_train,lam):
    """
    Muy parecido al entrena pero recibe un valor escalar lamba que podemos llamar el coefficiente de complejidad del modelo
    
    """
    eta = 0.1#La constante de aprendizaje
    observaciones = len(X_train)
    ### Para cada una de mis observaciones
    for i in range(observaciones):
        ### evaluo mi ygorro para cada vector de variables Xi
        sal = salida(w0,w,X_train.iloc[i,0:len(X_train.columns)])
        #Calculo el error de prediccion        
        error = Y_train.iloc[i] - sal
        # Obtengo el error w0
        w0 = w0 + eta*error - lam*w0
        columnasdatos = len(X_train.columns)
        ### Para cada uno de mis atributos/variables ajusto mis coeficientes incluyendo una regularización cuadratica       
        for j in range(columnasdatos):
            w[j] = w[j] + eta*error*X_train.iloc[i,j] - lam*w[j]
    return w0,w
"""
Queda una pregunta importante, que lambda debo de utilizar para regularizar el modelo?
Una alternativa es aprender esta lambda de mi set de datos
Para esto separo mi conunto de datos de entrenamiento en un set de validacion y uno de entrenamiento
"""
    
### Separo mi muestra en dos un set de entrenamiento y uno de validacion
X_train2, X_validate, Y_train2, Y_validate = train_test_split(X_train,Y_train, train_size=0.75)





### Genero una serie de lambdas posibles y pruebo 
lam = np.arange(-1,1,0.05)

# Vector para almacenar mis errores cuadraticos medios con cada una de las lambdas
error_lambda = []
### Para cada lambda obtengo una serie de w's que pruebo sobre mi conjunto de validación
for l in range(lam.size):
    #Inicializo mis coeficientes de manera aleatoria entre 0 y 1    
    w0_ini = rnd.random()
    w_ini = [rnd.random() for i in xrange(X_train2.shape[1])]
    ### Obtengo las w's en base a mi modelo de gradiente descendiente con regularización de Ridge
    w0_fin, w_fin = entrena_regularizado(w0_ini,w_ini,X_train2,Y_train2,lam[l])
    ### Calculo el error cuadratico medio con mi conjunto de validación
    error_lambda = error_lambda + [np.mean((Y_validate - (w0_fin + np.dot(X_validate,w_fin)))**2)]
    
    
#Para cada lambda reviso mi error cuadratico medio
plt.plot(lam,error_lambda)


#### Cual es el error cuadratico medio minimo?
min(error_lambda)
#### Cual es la lambda que minimiza el error cuadratico medio?
lam[[i for i,x in enumerate(error_lambda) if x == min(error_lambda)]][0]


"""
Llamo al metodo de gradiente descendiente con regularización con la lambda que minimiza el error cuadratico medio en el conjunto de validación
"""
w0_ini = rnd.random()
w_ini = [rnd.random() for i in xrange(X_train2.shape[1])]
w0_fin, w_fin = entrena_regularizado(w0_ini,w_ini,X_train2,Y_train2,lam[[i for i,x in enumerate(error_lambda) if x == min(error_lambda)]][0])

### Calculo mis y predecidas
Y_Predicted = w0_fin + np.dot(X_train2,w_fin)
### El error cuadratico medio es
np.mean((Y_train2 - Y_Predicted )**2)

### Visualizo mis predecidas y mis reales
for f in range(X_train2.shape[1]):
    plt.figure(f+1)
    plt.scatter(X_train2.iloc[0:X_train2.shape[0],f],Y_train2)
    plt.scatter(X_train2.iloc[0:X_train2.shape[0],f],Y_Predicted, color='red')
    plt.axis([-2, 2.5, -2, 4])
    plt.show()


#### Escalo mi conjunto de prueba usando los valores que obtuve de entrenamiento, ojo no debo de volver a calcular parametros sobre mi conjunto de pruebas
X_test = pd.DataFrame(data = ScaleX.transform(X_test), columns = ["X","X2","X3","X4"])
Y_test = pd.Series(data = ScaleY.transform(Y_test), name = 'y' )
### Calculo las y predecidas de mi conjunto de prueba
Y_Predicted_test = w0_fin + np.dot(X_test,w_fin)
### Y su error cuadratico medio
np.mean((Y_test - Y_Predicted_test )**2)


### VIsualizo mis predicciones versus mis variables reales
for f in range(X_test.shape[1]):
    plt.figure(f+1)
    plt.scatter(X_test.iloc[0:X_test.shape[0],f],Y_test)
    plt.scatter(X_test.iloc[0:X_test.shape[0],f],Y_Predicted_test, color='red')
    plt.show()
    
### Comparo mis predicciones y mis variables reales

fig, ax = plt.subplots()
ax.scatter(Y_test,Y_Predicted_test)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims)
ax.set_xlim(lims)
ax.set_ylim(lims)
    
#### Calculo mis errores de prediccion 
errores = Y_Predicted_test - Y_test

### Visualizo mis errores contra cada variable X
for f in range(X_test.shape[1]):
    plt.figure(f+1)
    plt.scatter(X_test.iloc[0:X_test.shape[0],f],errores)
    z = np.polyfit(X_test.iloc[0:len(X_test),f], errores, 1)
    p = np.poly1d(z)
    plt.plot(X_test.iloc[0:len(X_test),f],p(X_test.iloc[0:len(X_test),f]))
    plt.show()
### VIsualizo mis errores contra la variable dependiente
plt.scatter(Y_test,errores)
### VIsualizo la función de densidad de mis y reales y predecidas
df = pd.DataFrame({'predicted': Y_Predicted_test, 'real': Y_test})
df.plot(kind = "density")
### Como se distribuyen mis errores? Esperaria N(0,1)
errores.plot(kind = "density")

errores.describe()

### Guardo el RMS error del modelo regularizado
errores_modelos.iloc[0,1] = math.sqrt(np.mean((Y_test - Y_Predicted_test )**2))
#Comparo ambos valores
errores_modelos.plot(kind="bar")