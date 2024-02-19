import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy

filepath_1 = '/Users/samfuller/Desktop/545/FinTech545_sf-1/week04/DailyPrices.csv'
filepath_2 = '/Users/samfuller/Desktop/545/FinTech545_sf-1/week04/portfolio.csv'
prices = pd.read_csv(filepath_1)
portfolios = pd.read_csv(filepath_1)

def ewCovar(X, lam):
    X.set_index(X.columns[0], inplace=True)
    m, n = X.shape
    w = np.empty(m)
    
    # Remove the mean from the series
    X_mean = np.mean(X, axis=0)
    X = X - X_mean
    
    # Calculate weight. Realize we are going from oldest to newest
    for i in range(m):
        w[i] = (1 - lam) * lam**(m - i - 1)
    
    # Normalize weights to 1
    w /= np.sum(w)
    
    # Covariance calculation
    weighted_X = (w[:, np.newaxis] * X)  # Elementwise multiplication
    cov_matrix = np.dot(weighted_X.T, X)
    
    return cov_matrix

# 
