# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:09:18 2015

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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import sklearn.cross_validation as cross_validation
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


os.getcwd()
os.chdir("C:\\Users\\Stephane\\Desktop\\ITAM\\MachinLearning\\10_Clusters")
os.listdir(".")


abalone = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",header=None)
features = ['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','rings']
abalone.head()
'''
	Name		Data Type	Meas.	Description
	----		---------	-----	-----------
	Sex		nominal			M, F, and I (infant)
	Length		continuous	mm	Longest shell measurement
	Diameter	continuous	mm	perpendicular to length
	Height		continuous	mm	with meat in shell
	Whole weight	continuous	grams	whole abalone
	Shucked weight	continuous	grams	weight of meat
	Viscera weight	continuous	grams	gut weight (after bleeding)
	Shell weight	continuous	grams	after being dried
	Rings		integer			+1.5 gives the age in years
'''

abalone.columns = features

g = sns.PairGrid(abalone, hue="sex")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();

abalone_y =  pd.DataFrame(zip(abalone.iloc[:,0],abalone.iloc[:,0].astype('category').cat.codes),columns=["sex_cat","sex_num"])

abalone_unsupervised = abalone.iloc[:,1:abalone.shape[1]]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(abalone_unsupervised, abalone_y['sex_num'], train_size=0.80)
scaleX = StandardScaler()
scaleX.fit(X_train)
X_train=scaleX.transform(X_train)
X_test=scaleX.transform(X_test)
'''Decision Tree
'''
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
predict = clf.predict(X_test)
predict_proba = clf.predict_proba(X_test)

cm = confusion_matrix(y_test,predict)
cm
round(accuracy_score(y_test, predict),2)
##0.48
'''RandomTrees'''
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
y_predict = rf.predict(X_test)
cm = confusion_matrix(y_test,y_predict)
cm
round(accuracy_score(y_test, y_predict),2)
#0.53
'''
AdaBoost
'''
bdt_real = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1)
bdt_real.fit(X_train, y_train)
y_predict = bdt_real.predict(X_test)
cm = confusion_matrix(y_test,y_predict)
cm
round(accuracy_score(y_test, y_predict),2)
#0.51


'''Preprocesamiento con K Means'''
X_train, X_test, y_train, y_test = cross_validation.train_test_split(abalone_unsupervised, abalone_y['sex_num'], train_size=0.80)
scaleX = StandardScaler()
scaleX.fit(X_train)
X_train=scaleX.transform(X_train)
X_test=scaleX.transform(X_test)

n_cluster=3
km = KMeans(n_clusters=n_cluster,verbose=1)
km.fit(X_train)
#labels = km.labels_
#cluster_centers = km.cluster_centers_
y_train_cluster = km.predict(X_train)
y_test_cluster = km.predict(X_test)



trees = []
for i in range(n_cluster):
    '''Decision Tree
    '''
    ix = (y_train_cluster == i)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train[ix,:], y_train[ix])
    trees.append(clf)



y_predict_trees = np.full((len(y_test)), np.nan) 
for i in range(len(trees)):
    atree = trees[i]
    ix = (y_test_cluster == i)
    y_predict_trees[ix] = atree.predict(X_test[ix,:])

cm = confusion_matrix(y_test,y_predict_trees)
cm
round(accuracy_score(y_test, y_predict_trees),2)    
 #0.47
    
'''RandomTrees'''
rfs = []
for i in range(n_cluster):
    ix = (y_train_cluster == i)
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train[ix,:], y_train[ix])
    rfs.append(clf)

y_predict_rfs = np.full((len(y_test)), np.nan) 
for i in range(len(trees)):
    arf = rfs[i]
    ix = (y_test_cluster == i)
    y_predict_rfs[ix] = arf.predict(X_test[ix,:])

cm = confusion_matrix(y_test,y_predict_rfs)
cm
round(accuracy_score(y_test, y_predict_rfs),2) 
##0.56
  
   
'''
AdaBoost
'''
adas = []
for i in range(n_cluster):
    ix = (y_train_cluster == i)
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1)
    clf = clf.fit(X_train[ix,:], y_train[ix])
    adas.append(clf)

y_predict_adas = np.full((len(y_test)), np.nan) 
for i in range(len(trees)):
    aada = adas[i]
    ix = (y_test_cluster == i)
    y_predict_adas[ix] = aada.predict(X_test[ix,:])

cm = confusion_matrix(y_test,y_predict_adas)
cm
round(accuracy_score(y_test, y_predict_adas),2)    
#0.55
    
  