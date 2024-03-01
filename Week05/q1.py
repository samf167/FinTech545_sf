import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import t
import scipy
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.integrate import quad
import warnings

warnings.filterwarnings('ignore')

filepath = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Week05/problem1.csv'
x_df = pd.read_csv(filepath)
# print(x_df)
trials = 10000

alpha = 0.05 # set alpha
z_score = norm.ppf(alpha) # get inv cdf value
sdev = x_df['x'].std() # get stdev for sample set
column_mean = x_df['x'].mean()
#column_mean = 0

def ES(mu, sigma, alpha):
    phi_inv_alpha = norm.ppf(alpha)
    
    f_phi_inv_alpha = norm.pdf(phi_inv_alpha)
    
    ES = -mu + sigma * (f_phi_inv_alpha / alpha)
    return ES

# ---------------------------------------------
# Normal distribution with EW variance
def ewCovar(X, lam):
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

cov_matrix = ewCovar(x_df, 0.97) # get cov matrix with function
ew_Var = cov_matrix[0]
ew_sdev = ew_Var**0.5
VaR_norm_ewV = -(z_score * ew_sdev + column_mean) # get VaR

print("Normal VaR (EWV):", VaR_norm_ewV)
print("Normal ES (EWV):", ES(column_mean, ew_sdev, alpha))

# ---------------------------------------------
# MLE Fitted T
params = scipy.stats.t.fit(x_df, method='MLE') # Fit T distribution
alpha_return = scipy.stats.t.ppf(alpha, params[0], loc = params[1], scale =params[2]) # Save Df
VaR_mle_t = (-alpha_return) # calc VaR
print("T VaR:", VaR_mle_t)

integral, error = quad(lambda x: x * t.pdf(x, params[0], loc = params[1], scale =params[2]), -np.inf, -VaR_mle_t)

# Calculate ES
ES = -1/alpha * integral

print("T ES:", ES)

# ---------------------------------------------
# Historic Simulation
samples_list = [] # initialize sample list

# Bootstrap 10000 values from the historical x_df to approx distribution
for i in range(trials):
    new_samples = x_df.sample(n=10, replace=True) 
    samples_list.append(new_samples)

sample_draws = pd.concat(samples_list, ignore_index=True)

# Get alpha percentile for VaR
VaR_historic = -np.percentile(sample_draws, alpha*100)
print("Historic VaR:", VaR_historic)

# Calculate Es by brute force
total = 0
count = 0
val = 0
for i in range(trials*10):
    val = sample_draws.loc[i, 'x']
    if val < -VaR_historic:
        count+=1
        total= total + val

ES_historic = -total/count
print("Historic ES:", ES_historic)


