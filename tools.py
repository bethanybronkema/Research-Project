#functions to use in randomforest.py

import numpy as np
import matplotlib.pyplot as plt

def scatter_3d_data(data1: np.ndarray, data2: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c='k')
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c='b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

