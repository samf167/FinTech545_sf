import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy

filepath_1 = '/Users/samfuller/Desktop/545/FinTech545_sf-1/week04/DailyPrices.csv'
filepath_2 = '/Users/samfuller/Desktop/545/FinTech545_sf-1/week04/portfolio.csv'
prices = pd.read_csv(filepath_1)
portfolios = pd.read_csv(filepath_2)

final_row_df = prices.iloc[[-1]].transpose()
final_row_df.reset_index(level=0, inplace=True)
#final_row_df.rename(columns={'index': 'Stock'}, inplace=True)  #
portfolios = pd.merge(portfolios, final_row_df, left_on='Stock', right_on='index')

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

portfolio_A = portfolios[portfolios['Portfolio'] == 'A']
portfolio_B = portfolios[portfolios['Portfolio'] == 'B']
portfolio_C = portfolios[portfolios['Portfolio'] == 'C']

#portfolio_A.to_csv('output.csv', index=False, encoding='utf-8-sig') # DEBUGGER

ewCovar(prices, 0.94)

returns = (return_calculate(prices, method="DISCRETE", date_column = "Date"))

port_value = 0
port_value_A = 0
port_value_B = 0
port_value_C = 0

port_list = [portfolio_A, portfolio_B, portfolio_C]

for row in range(len(portfolio_A['index'])):
    holding_value = portfolio_A.iloc[row, 2] * portfolio_A.iloc[row, 4]
    portfolio_A.at[row, 'holding_value'] = holding_value
    port_value_A += holding_value

print(portfolio_A)
# for row in range(len(portfolio_A['index'])):



for row in range(len(portfolio_B['index'])):
    port_value_B = port_value_B + portfolio_B.iloc[row, 2]* portfolio_B.iloc[row, 4]

for row in range(len(portfolio_C['index'])):
    port_value_C = port_value_C + portfolio_C.iloc[row, 2]* portfolio_C.iloc[row, 4]


# Total Portfolio Value
for port in port_list:
    for row in range(len(port['index'])):
        port_value = port_value + port.iloc[row, 2]* port.iloc[row, 4]

print(port_value)






# Monte Carlo method