import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy

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

# Example usage:
# Assuming 'df' is a pandas DataFrame with price data and a 'date' column.
# returns_df = return_calculate(df, method="DISCRETE", date_column="date")

returns = (return_calculate(prices, method="DISCRETE", date_column = "Date"))
copy_returns = returns

column_mean = returns['META'].mean()

meta_returns = returns[['Date', 'META']]
meta_returns['META'] = returns['META'] - column_mean

meta_px = prices['META'].tail(n=1).values
#print(meta_px)

# Given we hold 1 meta share we multiply today's share price by all the returns in meta_returns to give us the potentia
#meta_returns['META_px'] = (1+meta_returns['META'])*meta_px
#print(meta_returns)


# Normal dsitribution estimation
sdev = meta_returns['META'].std()
alpha = 0.05
z_score = norm.ppf(alpha)

VaR_norm = -(z_score * sdev)*meta_px
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
cov_matrix = ewCovar(returns, 0.94)
meta_ewVar = cov_matrix[meta_idx,meta_idx]
ew_sdev = meta_ewVar**0.5
VaR_norm_ewV = -(z_score * ew_sdev)*meta_px
print("Normal VaR (EWV):", VaR_norm_ewV)


# MLE Fitted T
params = scipy.stats.t.fit(meta_returns['META'], method='MLE')
alpha_return = scipy.stats.t.ppf(alpha, params[0])
VaR_mle_t = -alpha_return*meta_px*sdev
print("MLE T VaR:", VaR_mle_t)

# AR(1) model

# Historic Simulation
sample_draws = meta_returns['META'].sample(n=100)
sample_px = (sample_draws)*meta_px
VaR_historic = -np.percentile(sample_px, 0.05)
print("Historic VaR:", VaR_historic)









