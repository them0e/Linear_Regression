
"""
Created on Fri Nov 15 23:41:09 2019

@author: them0e
"""



import numpy as np
import pandas as pd
import random as rd
"""
Hypothesis H(theta,x)
"""

def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(X.shape[0])
    return h

"""
Batch Gradient Descent 
"""
def BGD(theta, neta, num_iters, h, X, y, n):
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (neta/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            theta[j] = theta[j] - (neta/X.shape[0]) * sum((h-y) * X.transpose()[j])
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y)) # Cost function
    theta = theta.reshape(1,n+1)
    return theta, cost

"""
linear_regression(): It is the main function which takes the features matrix (X)
results verctor (y), learning rate (neta) and number of iterations as inputs
and outputs the final optimized theta i.e., the values of [theta_0, theta_1, theta_2, theta_3,….,theta_n]
for which the cost function almost achieves minima following Batch Gradient Descent
and cost which stores the value of cost for every iteration.
"""
def linear_regression(X, y, neta, num_iters):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    # initializing the parameter vector...
    theta = np.zeros(n+1)
    # hypothesis calculation....
    h = hypothesis(theta, X, n)
    # returning the optimized parameters by Gradient Descent...
    theta, cost = BGD(theta,neta,num_iters,h,X,y,n)
    return theta, cost


# Preprocessing Input data
data = pd.read_csv('heart.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, 13]

Sample1 = data.sample(frac=0.8)

# Feature scaling using mean and variance
#for j in range(13):
 #   for i in range(303):
  #      X[i,j] = (X[i,j]  - X[:,j].mean())/X[:,j].std()

# another feature scaling method
X = data.iloc[:, :-1].values
mean = np.ones(X.shape[1])
std = np.ones(X.shape[1])
for i in range(0, X.shape[1]):
    mean[i] = np.mean(X.transpose()[i])
    std[i] = np.std(X.transpose()[i])
    for j in range(0, X.shape[0]):
        X[j][i] = (X[j][i] - mean[i])/std[i]
  


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

        

theta, cost = linear_regression(X_train, y_train,0.009, 3000)

import matplotlib.pyplot as plt
cost = list(cost)
n_iterations = [x for x in range(1,3001)]
plt.plot(n_iterations, cost)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')

# Getting the predictions...
X_train = np.concatenate((np.ones((X_train.shape[0],1)), X_train)
                         ,axis = 1)
predictions = hypothesis(theta, X_train, X_train.shape[1] - 1)

for i in range(len(predictions)):
    if predictions[i] > 0.5:
           predictions[i] = 1 
    elif predictions[i] < 0.5: 
           predictions[i]=0
    
confusion_matrix = pd.crosstab(y_train, predictions, rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)
