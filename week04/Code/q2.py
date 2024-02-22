import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import warnings

warnings.filterwarnings('ignore')

filepath = '/Users/samfuller/Desktop/545/FinTech545_sf-1/week04/DailyPrices.csv'
prices = pd.read_csv(filepath)
trials = 100000

def return_calculate(prices, method="DISCRETE", date_column="date"):
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame: {prices.columns}")

    # Exclude the date column from the calculation
    vars = prices.columns.difference([date_column])

    # Extract the price data excluding the date column
    p = prices[vars].values

    # Calculate the simple returns or log returns
    if method.upper() == "DISCRETE":
        # Calculate simple returns
        returns = p[1:] / p[:-1] - 1
    elif method.upper() == "LOG":
        # Calculate log returns
        returns = np.log(p[1:] / p[:-1])
    else:
        raise ValueError(f"method: {method} must be in ('LOG', 'DISCRETE')")

    # Add the date column to the returns DataFrame
    returns_df = pd.DataFrame(data=returns, columns=vars)
    returns_df[date_column] = prices[date_column].iloc[1:].values

    # Reorder columns to have the date column at the beginning
    cols = [date_column] + [col for col in returns_df.columns if col != date_column]
    returns_df = returns_df[cols]

    return returns_df

# Initialize returns
returns = (return_calculate(prices, method="DISCRETE", date_column = "Date"))

# Center returns
column_mean = returns['META'].mean()
meta_returns = returns[['Date', 'META']]
meta_returns['META'] = returns['META'] - column_mean

# Get current share price
meta_px = prices['META'].tail(n=1).values

# Normal dsitribution estimation
sdev = meta_returns['META'].std() # get stdev for sample set
alpha = 0.05 # set alpha
z_score = norm.ppf(alpha) # get inv cdf value
VaR_norm = -((z_score * sdev + column_mean))*meta_px # get VaR
print("Normal VaR:", VaR_norm)

# Normal distribution with EW variance
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

meta_idx = returns.columns.get_loc('AAPL')-1 
cov_matrix = ewCovar(returns, 0.94) # get cov matrix with function
meta_ewVar = cov_matrix[meta_idx,meta_idx] # get meta var
ew_sdev = meta_ewVar**0.5
VaR_norm_ewV = -(z_score * ew_sdev + column_mean)*meta_px # get VaR
print("Normal VaR (EWV):", VaR_norm_ewV)

# MLE Fitted T
params = scipy.stats.t.fit(meta_returns['META'], method='MLE') # Fit T distribution
alpha_return = scipy.stats.t.ppf(alpha, params[0]) # Save Df
VaR_mle_t = (-alpha_return*sdev)*meta_px # calc VaR
print("MLE T VaR:", VaR_mle_t)

# AR(1) model

n_bootstraps = 100  # number of bootstrap samples
lags = 1  

bootstrap_estimates = np.zeros((n_bootstraps, lags + 1))
forecasts = []

for i in range(n_bootstraps):
    # Sample with replacement to create a synthetic dataset
    sample = meta_returns['META'].sample(n=len(meta_returns), replace=True)
    
    # Fit an AR(1) model to the synthetic dataset
    model = AutoReg(sample, lags=lags)
    model_fitted = model.fit()
    
    # Append the forecast to the forecasts list
    forecast_value = model_fitted.predict(start=len(sample), end=len(sample)).iloc[0]
    forecasts.append(forecast_value)
    
VaR_ar = -np.percentile(forecasts, 5)*meta_px

print("AR(1) VaR:", VaR_ar)


# Historic Simulation
samples_list = [] # initialize sample list

# Bootstrap 10000 values from the historical returns to approx distribution
for i in range(n_bootstraps):
    new_samples = meta_returns['META'].sample(n=100, replace=True) 
    samples_list.append(new_samples)

sample_draws = pd.concat(samples_list, ignore_index=True)
sample_px = sample_draws*meta_px

# Get alpha percentile for VaR
VaR_historic = -np.percentile(sample_px, 5)
print("Historic VaR:", VaR_historic)









