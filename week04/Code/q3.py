import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy
import warnings

warnings.filterwarnings('ignore')

# Import data
filepath_1 = '/Users/samfuller/Desktop/545/FinTech545_sf-1/week04/DailyPrices.csv'
filepath_2 = '/Users/samfuller/Desktop/545/FinTech545_sf-1/week04/portfolio.csv'
prices = pd.read_csv(filepath_1)
portfolios = pd.read_csv(filepath_2)

# Get most recent market prices and merge them into the portfolio dataframe
final_row_df = prices.iloc[[-1]].transpose()
final_row_df.reset_index(level=0, inplace=True)
portfolios = pd.merge(portfolios, final_row_df, left_on='Stock', right_on='index')

# Arithmetic return function
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

# EWC function
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

# Seperate seperate portfolios
portfolio_A = portfolios[portfolios['Portfolio'] == 'A']

portfolio_B = portfolios[portfolios['Portfolio'] == 'B']
portfolio_B.reset_index(drop=True, inplace=True)
portfolio_B.index = range(0, len(portfolio_B))

portfolio_C = portfolios[portfolios['Portfolio'] == 'C']
portfolio_C.reset_index(drop=True, inplace=True)
portfolio_C.index = range(0, len(portfolio_C))

#portfolio_A.to_csv('output.csv', index=False, encoding='utf-8-sig') # DEBUGGER

# Initialize returns
prices.reset_index(level=0, inplace=True)
returns = (return_calculate(prices, method="DISCRETE", date_column = "Date"))
returns = returns[returns.columns[:-1]]

# Initialize Values
port_value = 0
port_value_A = 0
port_value_B = 0
port_value_C = 0
span_value = 32.22 # lambda = 0.94
#span_value = 10000000000 # lambda = 0

# --------------------------------------------------------------------------

# Portfolio A

# Iterate through and calculate holding values and total portfolio value
for row in range(len(portfolio_A['index'])):
    holding_value = portfolio_A.iloc[row, 2] * portfolio_A.iloc[row, 4]
    portfolio_A.at[row, 'holding_value'] = holding_value
    port_value_A += holding_value

# Calculate delta
for row in range(len(portfolio_A['index'])):
    delta = portfolio_A.iloc[row, 5]/port_value_A
    portfolio_A.loc[row, 'delta'] = delta

# Check our delta makes sense
#print(portfolio_A['delta'].sum())

# Filter returns to include only stocks in portfolio A
port_A_stocks = portfolio_A['index'].tolist()
returns_A = returns.loc[:, returns.columns.isin(port_A_stocks)]

# Built in EW Formula calculation (active)
ew_cov_matrix = returns_A.ewm(span=span_value, min_periods=1, adjust=False).cov(pairwise=True)
sigma_A = ew_cov_matrix.loc[ew_cov_matrix.index[-1][0]]

# Function EW formula
returns_A.reset_index(level=0, inplace=True)
sigma_A_2 = ewCovar(returns_A, 0.94)

# Reshape delta
delta = portfolio_A['delta'].values.reshape(-1, 1)  # Reshape to (33, 1)

# Calculate p_sig and VaR
p_sig = (delta.T.dot(sigma_A).dot(delta))**(0.5)
p_sig_value = p_sig.item()
VaR_A = -port_value_A * norm.ppf(0.05)*p_sig_value

print("VaR_A", VaR_A)
print("Portfolio Value", port_value_A)

# --------------------------------------------------------------------------

# Portfolio B

# Iterate through and calculate holding values and total portfolio value
for row in range(len(portfolio_B['index'])):
    holding_value = portfolio_B.iloc[row, 2] * portfolio_B.iloc[row, 4]
    portfolio_B.at[row, 'holding_value'] = holding_value
    port_value_B += holding_value

# Calculate delta

for row in range(len(portfolio_B['index'])):
    delta = portfolio_B.iloc[row, 5]/port_value_A
    portfolio_B.loc[row, 'delta'] = delta

# Check our delta makes sense
# print(portfolio_B['delta'].sum())

# Filter returns to include only stocks in portfolio B
port_B_stocks = portfolio_B['index'].tolist()
returns_B = returns.loc[:, returns.columns.isin(port_B_stocks)]

# Built in EW Formula (active)
ew_cov_matrix = returns_B.ewm(span=span_value, min_periods=1, adjust=False).cov(pairwise=True)
sigma_B = ew_cov_matrix.loc[ew_cov_matrix.index[-1][0]]

# Defined EW function
returns_B.reset_index(level=0, inplace=True)
sigma_B_2 = ewCovar(returns_B, 0.94)

# Reshape delta and calculate VaR
delta = portfolio_B['delta'].values.reshape(-1, 1)  # Reshape to (33, 1)
p_sig = (delta.T.dot(sigma_B).dot(delta))**(0.5)
p_sig_value = p_sig.item()
VaR_B = -port_value_B * norm.ppf(0.05)*p_sig_value

print("VaR_B", VaR_B)
print("Portfolio Value", port_value_B)

# --------------------------------------------------------------------------

# Portfolio C

# Iterate through and calculate holding values and total portfolio value
for row in range(len(portfolio_C['index'])):
    holding_value = portfolio_C.iloc[row, 2] * portfolio_C.iloc[row, 4]
    portfolio_C.at[row, 'holding_value'] = holding_value
    port_value_C += holding_value

# Calculate Delta
for row in range(len(portfolio_C['index'])):
    delta = portfolio_C.iloc[row, 5]/port_value_A
    portfolio_C.loc[row, 'delta'] = delta

# Check our delta makes sense
# print(portfolio_C['delta'].sum())

# Filter returns for stocks in C
port_C_stocks = portfolio_C['index'].tolist()
returns_C = returns.loc[:, returns.columns.isin(port_C_stocks)]

# Built in EW Formula (active)
ew_cov_matrix = returns_C.ewm(span=span_value, min_periods=1, adjust=False).cov(pairwise=True)
sigma_C = ew_cov_matrix.loc[ew_cov_matrix.index[-1][0]]

# Defined EW
returns_C.reset_index(level=0, inplace=True)
sigma_C_2 = ewCovar(returns_C, 0.94)

# Calculate delta and VaR_C
delta = portfolio_C['delta'].values.reshape(-1, 1)  # Reshape to (33, 1)
p_sig = (delta.T.dot(sigma_C).dot(delta))**(0.5)
p_sig_value = p_sig.item()

VaR_C = -port_value_C * norm.ppf(0.05)*p_sig_value

print("VaR_C", VaR_C)
print("Portfolio Value", port_value_C)

# --------------------------------------------------------------------------

# Portfolio TTL

# Iterate through and calculate holding values and total portfolio value
for row in range(len(portfolios['index'])):
    holding_value = portfolios.iloc[row, 2] * portfolios.iloc[row, 4]
    portfolios.at[row, 'holding_value'] = holding_value
    port_value += holding_value

# Calculate Delta
for row in range(len(portfolios['index'])):
    delta = portfolios.iloc[row, 5]/port_value_A
    portfolios.loc[row, 'delta'] = delta

# Check our delta makes sense
# print(portfolios['delta'].sum())

# Include all stocks in portfolio
port_ttl_stocks = portfolios['index'].tolist()
returns_ttl = returns.loc[:, returns.columns.isin(port_ttl_stocks)]

# Built in EW Formula (active)
ew_cov_matrix = returns_ttl.ewm(span=span_value, min_periods=1, adjust=False).cov(pairwise=True)
sigma_ttl = ew_cov_matrix.loc[ew_cov_matrix.index[-1][0]]

# Defined EW
returns_ttl.reset_index(level=0, inplace=True)
sigma_ttl_2 = ewCovar(returns_ttl, 0.94)

# Calc delta and VaR
delta = portfolios['delta'].values.reshape(-1, 1)  # Reshape to (33, 1)
p_sig = (delta.T.dot(sigma_ttl).dot(delta))**(0.5)
p_sig_value = p_sig.item()
VaR_ttl = -port_value * norm.ppf(0.05)*p_sig_value

print("VaR_Total", VaR_ttl)
print("Portfolio Value", port_value)

