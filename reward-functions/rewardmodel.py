import os.path
import numpy as np
from numpy import savetxt, loadtxt


def generateXvector(X):
    """ Taking the original independent variables matrix and add a row of 1 which corresponds to x_0
        Parameters:
          X:  independent variables matrix
        Return value: the matrix that contains all the values in the dataset, not include the outcomes variables.
    """
    s = (len(X), 1)
    vectorX = np.c_[np.ones(s), X]

    return vectorX


def theta_init(X):
    """ Generate an initial value of vector θ from the original independent variables matrix
         Parameters:
          X:  independent variables matrix
        Return value: a vector of theta filled with initial guess
    """
    theta = np.random.randn(len(X[0])+1, 1)
    return theta


def Multivariable_Linear_Regression(X, y, learningrate, iterations):
    y_new = np.reshape(y, (len(y), 1))
    cost_lst = []
    vectorX = generateXvector(X)
    theta = theta_init(X)
    m = len(X)
    for i in range(iterations):
        gradients = 2/m * vectorX.T.dot(vectorX.dot(theta) - y_new)
        theta = theta - learningrate * gradients
        y_pred = vectorX.dot(theta)
        cost_value = 1/(2*len(y))*((y_pred - y)**2)
        # Calculate the loss for each training instance
        total = 0
        for ii in range(len(y)):
            total += cost_value[ii][0]
            # Calculate the cost function for each iteration
        cost_lst.append(total)
    savetxt('theta.csv', theta)
    return theta, vectorX


def calculate_reward(feature, h, activity):
    reward = 0
    if not activity == 0:
        theta, x = Multivariable_Linear_Regression(feature, h, 0.01, 100)
        m = len(x)
        for i in range(len(theta)):
            reward += x[m - 1][i] * theta[i][0]
        return reward
    elif os.path.exists('theta.csv'):
        theta = loadtxt('theta.csv')
        x = generateXvector(feature)
        m = len(x)
        for i in range(len(x[0])):
            reward += x[m][i] * theta[i+1]
        return reward
    else:
        return 0
