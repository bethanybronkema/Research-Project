#functions to use in randomforest.py

import numpy as np
import pandas as pd

def standardize(X: pd.DataFrame) -> pd.DataFrame:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    x_hat = np.empty((X.shape[0], X.shape[1]))
    for i in range(1, (X.shape[0]+1)):
        for j in range(X.shape[1]):
            x_hat.loc[i, j] = (X.loc[i, j] - np.mean(X.loc[:, j]))/np.std(X.loc[:, j])
    return x_hat

