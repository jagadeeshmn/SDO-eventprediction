#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:33:51 2018

@author: priyashrivastava
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# X = pd.read_csv('X.csv', header=0)
# Y = pd.read_csv('Y.csv', header=0)


def Perform_Logistic_Regression(X,Y):
	X = X.T
	Y = Y.T
	print(X.shape)
	print(Y.shape)	
	x_train ,x_test = train_test_split(X,test_size=0.7) 
	y_train ,y_test = train_test_split(Y,test_size=0.7) 
	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)
	logreg = LogisticRegression(fit_intercept=True)
	for i in range(y_train.shape[0]):
		logreg.fit(x_train,y_train[:,i])
		print(logreg.coef_)
		print("COEFFICIENTS")
		print("_+_+_+_+_+_+")
# print(logreg.score(x_train,y_train))
print("_+_+_+_+_+_+")

"""def Load_data(data, at = 0.7):
    data =np.transpose(data)
    data = pd.DataFrame(data=data)
    n = data.shape[0]
    idx = np.random.shuffle(n)
    print(idx)
    train_idx = view(idx, 1:floor(Int, at*n)) 
    
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)"""


# x_train=np.split(X.values, [368639,7])
# x_train = (np.array(x_train))
# print (x_train.shape)


# def split_padded(X,n):
#     padding = (-len(X))%n
#     return np.split(np.concatenate((X,np.zeros(padding))),n)

# y_train=np.hsplit(Y, 2)
# print(y_train)

# Xdata = np.array(X)
# Ydata = np.array(Y)

# print (Xdata.shape)
# print (Ydata.shape)
# Xdata = Xdata.T
# Ydata = Ydata.T

# x_train ,x_test = train_test_split(Xdata,test_size=0.7) 

# y_train ,y_test = train_test_split(Ydata,test_size=0.7) 

# print (x_train.shape)
# print (x_test.shape)

# print (y_train.shape)
# print (y_test.shape)
