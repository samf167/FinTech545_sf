import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import spearmanr
import scipy
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.integrate import quad
import warnings

warnings.filterwarnings('ignore')

# Import data
filepath_1 = '/Users/samfuller/Desktop/545/FinTech545_sf-2/week05/DailyPrices.csv'
filepath_1 = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Week04/Data/DailyPrices.csv'
filepath_2 = '/Users/samfuller/Desktop/545/FinTech545_sf-2/week05/portfolio.csv'
filepath_2 = '/Users/samfuller/Desktop/545/FinTech545_sf-2/week04/Data/portfolio.csv'

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

# Seperate seperate portfolios
portfolio_A = portfolios[portfolios['Portfolio'] == 'A']

portfolio_B = portfolios[portfolios['Portfolio'] == 'B']
portfolio_B.reset_index(drop=True, inplace=True)
portfolio_B.index = range(0, len(portfolio_B))

portfolio_C = portfolios[portfolios['Portfolio'] == 'C']
portfolio_C.reset_index(drop=True, inplace=True)
portfolio_C.index = range(0, len(portfolio_C))

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
alpha = 0.05
n = 5000
#span_value = 10000000000 # lambda = 0

# --------------------------------------------------------------------------

# Create returns dists for all portfolios
port_A_stocks = portfolio_A['index'].tolist()
returns_A = returns.loc[:, returns.columns.isin(port_A_stocks)]
nma = returns_A.columns

port_B_stocks = portfolio_B['index'].tolist()
returns_B = returns.loc[:, returns.columns.isin(port_B_stocks)]

port_C_stocks = portfolio_C['index'].tolist()
returns_C = returns.loc[:, returns.columns.isin(port_C_stocks)]

port_ttl_stocks = portfolios['index'].tolist()
returns_ttl = returns.loc[:, returns.columns.isin(port_ttl_stocks)]
nms = returns_ttl.columns

portfolio_return = 0

fittedModels = pd.DataFrame()

uniform_returns = {}
standard_normal_returns = {}
deg_free = {}
loc_t = {}
scale_t = {}

# Process returns for portfolio A and B (T-distributed) and transform to uniform then standard normal
for returns_df in [returns_A, returns_B]:
    for col in returns_df.columns:
        deg_free[col], loc_t[col], scale_t[col] = scipy.stats.t.fit(returns_df[col], floc=0)
        uniform_returns[col] = scipy.stats.t.cdf(returns_df[col], deg_free[col], loc=0, scale=scale_t[col])
        standard_normal_returns[col] = scipy.stats.norm.ppf(uniform_returns[col])

# Process returns for portfolio C (Normally-distributed) and transform to uniform then standard normal
for col in returns_C.columns:
    scale = returns_C[col].std()
    uniform_returns[col] = scipy.stats.norm.cdf(returns_C[col], loc=0, scale=scale)
    standard_normal_returns[col] = scipy.stats.norm.ppf(uniform_returns[col])

uniform_returns_df = pd.DataFrame(uniform_returns)
standard_normal_returns_df = pd.DataFrame(standard_normal_returns)

# calcualte spearman correlation matrix of standard normal df
spear = scipy.stats.spearmanr(standard_normal_returns_df)
mean = np.zeros(len(spear))
cov = spear[0]

mean_vector = np.zeros(len(cov))  # Assuming a mean vector of zeros for simplicity

# simulate from the multivariate normla distribution using previous correaltion matrix
simulated_normals = np.random.multivariate_normal(mean_vector, cov, size=n)

simU = pd.DataFrame(norm.cdf(simulated_normals), columns=nms)

simulatedReturnsData = {}

# Transform back to uniform and then back to returns for both T and normally distributed variables
for col in returns_C.columns:
    std_dev_return = returns_C[col].std()
    simulatedReturnsData[col] = norm.ppf(simU[col], loc=0, scale=std_dev_return)

for returns_df in [returns_A, returns_B]:
    for col in returns_df.columns:
        simulatedReturnsData[col] = t.ppf(simU[col], deg_free[col],loc=0 , scale=scale_t[col])

simulatedReturns = pd.DataFrame(simulatedReturnsData)
#print(simulatedReturns)

# complete with model fitting
# Create a DataFrame for iterations
iterations = pd.DataFrame({'iteration': range(1, n)})

# Perform a cross join between Portfolio and iterations
values = pd.merge(portfolios.assign(key=1), iterations.assign(key=1), on='key').drop('key', axis=1)
print(values)
# Calculate current value, simulated value, and PnL using portfolio data
values['currentValue'] = values['Holding'] * values[265]
values['simulatedValue'] = values.apply(lambda x: x['currentValue'] * (1.0 + simulatedReturns.loc[x['iteration'], x['Stock']]), axis=1)
values['pnl'] = values['simulatedValue'] - values['currentValue']
# values.to_csv('merged_values.csv', index=False)

# group by iteration to get the monte carlo simulated values for pnl
gdf = values.groupby('iteration')
totalValues = gdf.agg({
    'currentValue': 'sum',
    'simulatedValue': 'sum',
    'pnl': 'sum'
}).reset_index()

print(totalValues)

# Take alpha percentile for var
VaR_historic = -np.percentile(totalValues['pnl'], alpha*100)
print("Total Portfolio VaR:", VaR_historic)

# Calculate Es 
total = 0
count = 0
val = 0
for i in range(n-1):
    val = totalValues.loc[i, 'pnl']
    if val < -VaR_historic:
        count+=1
        total= total + val

ES_historic = -total/count
print("Total Portfolio ES:", ES_historic)