import numpy as np
import scipy.stats as stats
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

# Read in data from filepath
filepath = "/Users/samfuller/Desktop/545/FinTech545_sf/Week02/problem2_x.csv"
data = np.genfromtxt(filepath, delimiter=',') 

# Chop first row of data off
data = data[1:]

# Calculate the mean vector and covariance matrix of sample data for sanity checking
mean_vector = np.mean(data, axis=0)
covariance_matrix = np.cov(data, rowvar=False)

print("Sample Means =", mean_vector)
print("Sample Covariance=", covariance_matrix)

# Create function to calculate negative log likelihood of multivariate normal distribuiton
def negative_log_likelihood(params, X):
    p = X.shape[1]  # dimensionality of data points
    mu = params[:p]  # mean vector
    Sigma = params[p:].reshape((p, p))  # covariance matrix
    m = X.shape[0]  # number of data points
    log_det_Sigma = np.log(np.linalg.det(Sigma)) # calculate log determinant of covariance matrix
    inv_Sigma = np.linalg.inv(Sigma) # calculate inverse of covariance matrix
    
    total = m * p * 0.5 * np.log(2 * np.pi) + 0.5 * m * log_det_Sigma # initialize log likelihood before sum term
    
    # Calculate summed term using for loop
    for i in range(m):
        diff = X[i, :] - mu # actual less mean
        total += 0.5 * np.dot(diff.T, np.dot(inv_Sigma, diff))
    
    return total # return log likelihood

# Initial guess parameters for MLS
initial_mu = [0, 1] # initialize basic mean values
initial_Sigma = np.eye(data.shape[1]) # identity matrix
initial_params = np.hstack([initial_mu, initial_Sigma.ravel()])  # flatten the parameters

# Perform minimization
result = minimize(negative_log_likelihood, initial_params, args=(data,), method='L-BFGS-B')
# Save results
fitted_params = result.x
fitted_mu = fitted_params[:data.shape[1]]
fitted_Sigma = fitted_params[data.shape[1]:].reshape((data.shape[1], data.shape[1]))
print("Fitted mean:", fitted_mu)
print("Fitted covariance matrix:", fitted_Sigma)

# Read in our observed x1 data from alternate datafile
observed = pd.read_csv("/Users/samfuller/Desktop/545/FinTech545_sf/Week02/problem2_x1.csv")
observed = observed["x1"].values 

# Calculate conditional distributions of X2 given each observed value using fitted parameters
mean_x1, mean_x2 = fitted_mu
var_x1 = fitted_Sigma[0, 0] # saving values from fitted distribution
var_x2 = fitted_Sigma[1, 1]
cov_x1_x2 = fitted_Sigma[0, 1]
conditional_means = mean_x2 + cov_x1_x2 / var_x1 * (observed - mean_x1)
conditional_variances = var_x2 - cov_x1_x2**2 / var_x1
conditional_variances = [conditional_variances] * len(conditional_means)

for i in range(len(observed)):
    print("X1 = ", observed[i], "X2 Mean = ", conditional_means[i], "X2 Variance =", conditional_variances[i])

# Calculate z scores and associated confidence interval for each expected value of X2
z_score = stats.norm.ppf(0.975)
margins = z_score * np.sqrt(conditional_variances)
lower_bounds = conditional_means - margins
upper_bounds = conditional_means + margins

# Plot expected value of X2 against observed X1 including confidence interval
plt.figure(figsize=(10, 6))
plt.plot(observed, conditional_means, 'o', label='Conditional Expectation of X2')
plt.fill_between(observed, lower_bounds, upper_bounds, color='gray', alpha=0.2, label='95% CI')
plt.xlabel('X1')
plt.ylabel('Expected Value of X2')
plt.legend()
plt.title('Expected Value of X2 with 95% Confidence Interval Given X1')
plt.grid(True)
plt.show()