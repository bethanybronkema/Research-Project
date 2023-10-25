#functions to use in randomforest.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score)

def forest_classifying(f_train, f_test, t_train, t_test, n_estimators, max_depth):

    # make the classifier
    forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=10)
    forest.fit(f_train, t_train)

    # classify test data
    predictions = forest.predict(f_test)

    # get accuracy, precision, recall, and confusion matrix of predictions
    accuracy = accuracy_score(t_test, predictions)
    precision = precision_score(t_test, predictions)
    recall = recall_score(t_test, predictions)
    con_matrix = confusion_matrix(t_test, predictions)

    return accuracy, precision, recall, con_matrix

def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    np.random.seed(1234)
    I = np.identity(k)
    sigma = np.square(var)*I
    X = np.random.multivariate_normal(mean, sigma, n)
    return X, mean, sigma

def scatter_3d_data(data1: np.ndarray, data2: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c='r')
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c='b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def chernoff(s, mu0, mu1, sigma0, sigma1):
    term1 = ((s*(1-s))/2)*((mu1 - mu0).T) @ np.linalg.inv(s*sigma0 + (1-s)*sigma1) @ (mu1 - mu0)
    num = np.linalg.det(s*sigma0 + (1-s)*sigma1)
    den = (np.power(np.linalg.det(sigma0), s) * np.power(np.linalg.det(sigma1), 1-s))
    term2 = (1/2)*np.log(num/den)
    distance = term1 + term2
    return np.exp(-distance)

def plot_chernoff_error(num: int, mu0, mu1, sigma0, sigma1):

    '''
    num: number of s values to test
    '''

    s = np.linspace(0, 1, num)
    y = np.zeros(num)
    for i in range(num):
        y[i] = chernoff(s[i], mu0, mu1, sigma0, sigma1)
    plt.plot(s, y)
    plt.show()