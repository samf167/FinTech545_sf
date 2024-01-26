import pandas as pd
import numpy as np
import statistics as st
import scipy
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
import seaborn
import math
import matplotlib.pyplot as plt

# Read in data
filepath = "/Users/samfuller/Desktop/545/FinTech545_sf/Week02/problem2.csv"
data = pd.read_csv(filepath)
x = data["x"].values 
plt_x = x # initialize for plot later on
y = data["y"].values

x = sm.add_constant(x)

# Complete regression and save optimal parameters
model = sm.OLS(y, x).fit()
parameters = model.params
intercept = parameters[0]
slope = parameters[1]
residual = model.resid

# Get OLS expected y values for plotting
y_ols = slope*x

print(model.summary())
beta = 0.77

# Formulate log likelihood using formula from notes
def ll(inputs):
    beta = inputs[1]
    std = inputs[0]
    y_hat = beta*x
    error = [0] * len(y_hat)
    ttl = 0

    for i in range(len(y_hat)):
        error[i] = y[i] - y_hat[i]
        error_zero = (error[i] - 0)**2
        ttl = ttl + error_zero
    stdev = std
    ll = -(len(x)/2)*(np.log((stdev**2*2*math.pi))) - ((1/(2*(stdev**2)))*ttl)
    ll = ll[1]
    return -ll

# Minimize the log likelihood
inputs = [1, beta]
mle = minimize(ll, x0 = inputs, method = "L-BFGS-B")
beta_hat = mle.x[1:]
y_mle = beta_hat*plt_x
y_ols = slope*plt_x

# Plot original data and OLS/MLE line of best fit
plt.clf()
plt.figure(figsize=(10, 6))
plt.plot(plt_x, y, 'o', label='Given data')
plt.plot(plt_x, y_mle, '-', label='MLE Fit line')
plt.plot(plt_x, y_ols, '-', label='OLS Fit line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('MLE vs. OLS Fit through Data')
plt.grid(True)
plt.show()