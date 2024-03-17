import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import t
import scipy
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.integrate import quad
import warnings
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

warnings.filterwarnings('ignore')

filepath = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Week05/problem1.csv'
x_df = pd.read_csv(filepath)
# print(x_df)
trials = 10000

alpha = 0.05 # set alpha
z_score = norm.ppf(alpha) # get inv cdf value
sdev = x_df['x'].std() # get stdev for sample set
column_mean = x_df['x'].mean()
column_mean = 0

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
VaR_norm_ewV = -(z_score * ew_sdev + column_mean) # get VaR and make it absolute by subtracting mean

print("Normal VaR (EWV):", VaR_norm_ewV)
print("Normal ES (EWV):", ES(column_mean, ew_sdev, alpha))

ES_norm = ES(column_mean, ew_sdev, alpha)[0]
VaR_norm_ewV = VaR_norm_ewV[0]

x = np.linspace(-0.3, 0.3, 1000)
pdf_fitted = stats.norm.pdf(x, column_mean, ew_sdev)
sns.histplot(x_df, kde=False, stat='density', color='skyblue', alpha=0.6, bins=30)

plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Normal Distribution')
plt.axvline(x=-ES_norm, color='green', linestyle='--', linewidth=2, label=f'ES = -{ES_norm:.2f}')
plt.axvline(x=-VaR_norm_ewV, color='green', linestyle='-', linewidth=2, label=f'VaR = -{VaR_norm_ewV:.2f}')

plt.title('Fitted Normal Distribution Overlaying Histogram of Data')
plt.xlabel('Returns')
plt.ylabel('Density')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
sns.despine(offset=10, trim=True) 

plt.show()

# ---------------------------------------------
# MLE Fitted T
params = scipy.stats.t.fit(x_df, method='MLE') # Fit T distribution
print(params)
alpha_return = scipy.stats.t.ppf(alpha, params[0], loc = params[1], scale =params[2]) # Save Df
VaR_mle_t = (-alpha_return) # calc VaR
print("T VaR:", VaR_mle_t)

# Calculate ES via the integral definition
integral, error = quad(lambda x: x * t.pdf(x, params[0], loc = params[1], scale =params[2]), -np.inf, -VaR_mle_t)

# Finish ES calculation
ES = -1/alpha * integral

print("T ES:", ES)

pdf_fitted_t = stats.t.pdf(x, *params)

sns.histplot(x_df, kde=False, stat='density', color='skyblue', alpha=0.6, bins=30)

plt.plot(x, pdf_fitted_t, 'r-', lw=2, label='Fitted T-distribution')
plt.axvline(x=-ES, color='green', linestyle='--', linewidth=2, label=f'ES = {-ES:.2f}')
plt.axvline(x=-VaR_mle_t, color='green', linestyle='-', linewidth=2, label=f'VaR = {-VaR_mle_t:.2f}')

plt.title('Fitted T-distribution Overlaying Histogram of Data')
plt.xlabel('Returns')
plt.ylabel('Density')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
sns.despine(offset=10, trim=True) 

plt.show()

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

sns.histplot(sample_draws, kde=False, stat='density', color='skyblue', alpha=0.6, bins=30)

#plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted ')
plt.axvline(x=-ES_historic, color='green', linestyle='--', linewidth=2, label=f'ES = {-ES_historic:.2f}')
plt.axvline(x=-VaR_historic, color='green', linestyle='-', linewidth=2, label=f'VaR = {-VaR_historic:.2f}')

plt.title('Histogram of Data Sampled from Historical Distribution')
plt.xlabel('Returns')
plt.xlim(-0.3, 0.3)
plt.xticks(np.arange(-0.3, 0.31, 0.1), ['-0.3', '-0.2', '-0.1', '0.0', '0.1', '0.2', '0.3'])
plt.ylabel('Density')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
sns.despine(offset=10, trim=True) 

plt.show()



# plotting all on one 
pdf_fitted = stats.norm.pdf(x, column_mean, ew_sdev)
sns.histplot(x_df, kde=False, stat='density', color='skyblue', alpha=0.6, bins=30)

plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Normal Distribution')
plt.plot(x, pdf_fitted_t, 'b-', lw=2, label='Fitted T-distribution')
plt.axvline(x=-ES_norm, color='red', linestyle='--', linewidth=2, label=f'Norm ES = -{ES_norm:.2f}')
plt.axvline(x=-VaR_norm_ewV, color='red', linestyle='-', linewidth=2, label=f'Norm VaR = -{VaR_norm_ewV:.2f}')
plt.axvline(x=-ES, color='blue', linestyle='--', linewidth=2, label=f'T ES = {-ES:.2f}')
plt.axvline(x=-VaR_mle_t, color='blue', linestyle='-', linewidth=2, label=f'T VaR = {-VaR_mle_t:.2f}')
plt.axvline(x=-ES_historic, color='green', linestyle='--', linewidth=2, label=f'ES = {-ES_historic:.2f}')
plt.axvline(x=-VaR_historic, color='green', linestyle='-', linewidth=2, label=f'VaR = {-VaR_historic:.2f}')


plt.title('Fitted Normal and T Distributions Overlaying Histogram of Data')
plt.xlabel('Returns')
plt.ylabel('Density')
plt.xlim(-0.3, 0.3)
plt.xticks(np.arange(-0.3, 0.31, 0.1), ['-0.3', '-0.2', '-0.1', '0.0', '0.1', '0.2', '0.3'])
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
sns.despine(offset=10, trim=True) 

plt.show()