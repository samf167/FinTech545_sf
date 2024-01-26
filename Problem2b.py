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

# read in data
data = pd.read_csv("/Users/samfuller/Desktop/545/FinTech545_sf/Week02/problem2.csv")
x = data["x"].values
# save x without constant for plots later
plt_x = x
y = data["y"].values

x = sm.add_constant(x)

# compute regression and save parameters
model = sm.OLS(y, x).fit()
parameters = model.params
intercept = parameters[0]
slope = parameters[1]
residual = model.resid

print(model.summary())

# define a beta for our MLE intial point
beta = 0.77

# Function computes MLE based on T distribution of errors
def ll(inputs):
    beta = inputs[1]
    std = inputs[0]
    y_hat = beta*x
    error = [0] * len(y_hat)
    ttl = 0 # initialize sum term
    k = len(x)-1 # degrees of freedom

    for i in range(len(y_hat)):
        error[i] = y[i] - y_hat[i]
        error_zero = (error[i] - 0)**2
        sum_arg = np.log(k+error_zero)
        ttl = ttl + sum_arg

    ll = -ttl
    ll = ll[1]
    return -ll

# Minimize the log likelihood and print the outputted parameters
inputs = [1, beta]
mle = minimize(ll, x0 = inputs, method = "L-BFGS-B")
print(mle)
print("Beta =", mle.x[1])
print("Sd = ", mle.x[0])

beta = mle.x[1]
# Compute estimated y values from our MLE beta
y_mle_t = beta*plt_x

# Plot results 
plt.clf()
plt.figure(figsize=(10, 6))
plt.plot(plt_x, y, 'o', label='Given data')
plt.plot(plt_x, y_mle_t, '-', label='MLE T-dist Fit line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('MLE T-dist through Data')
plt.grid(True)
plt.show()