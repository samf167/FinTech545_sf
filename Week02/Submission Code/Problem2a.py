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

# Complete regression and save optimal parameters for y = bx
model = sm.OLS(y, x).fit()
parameters = model.params
slope = parameters[0]
residual = model.resid
print(np.std(residual))

# Get OLS expected y values for plotting
y_ols = slope*x

print(model.summary()) # print OLS summary
beta = 0.70 # initialize guess beta

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
    return -ll

# Minimize the log likelihood
inputs = [1, beta]
mle = minimize(ll, x0 = inputs, method = "L-BFGS-B")
beta_hat = mle.x[1:]
y_mle = beta_hat*plt_x
y_ols = slope*plt_x

print(mle)

# Plot original data and OLS line of best fit
'''plt.clf()
plt.figure(figsize=(10, 6))
plt.plot(plt_x, y, 'o', label='Given data')
#plt.plot(plt_x, y_mle, '-', label='MLE Fit line')
plt.plot(plt_x, y_ols, '-', label='OLS Fit line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('OLS Fit through Data')
plt.grid(True)
plt.show()'''

# Plot original data and MLE line of best fit
plt.clf
plt.figure(figsize=(10, 6))
plt.plot(plt_x, y, 'o', label='Given data')
plt.plot(plt_x, y_mle, '-', label='MLE Fit line')
#plt.plot(plt_x, y_ols, '-', label='OLS Fit line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('MLE vs. Fit through Data')
plt.grid(True)
plt.show()

# calculate rsq of the MLE T dist model
corr_matrix = np.corrcoef(y, y_ols)
corr = corr_matrix[0,1]
R_sq = corr**2

print(R_sq)