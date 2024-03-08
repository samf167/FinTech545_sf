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
import functions as fn
import math
import csv
from sklearn.decomposition import PCA
from functions import _getAplus
from functions import _getPS  
from functions import wgtNorm


# Import data
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test9_1_portfolio.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test9_1_returns.csv'
returns = pd.read_csv(b)
portfolios = pd.read_csv(a)

returns_A = returns['A']
returns_B = returns['B']

# Get most recent market prices and merge them into the portfolio dataframe
final_row_df = portfolios[['Stock', 'Starting Price','Holding']]
final_row_df.reset_index(level=0, inplace=False)


# Initialize Values
port_value = 0
port_value_A = 0
port_value_B = 0
port_value_C = 0
alpha = 0.05
n = 5000
#span_value = 10000000000 # lambda = 0

# --------------------------------------------------------------------------
portfolio_return = 0

fittedModels = pd.DataFrame()

uniform_returns = {}
standard_normal_returns = {}
deg_free = {}
loc_t = {}
scale_t = {}

# Process returns for portfolio A and B (T-distributed)
deg_free['B'], loc_t['B'], scale_t['B'] = scipy.stats.t.fit(returns['B'], floc=0)
uniform_returns['B'] = scipy.stats.t.cdf(returns['B'], deg_free['B'], loc=0, scale=scale_t['B'])
standard_normal_returns['B'] = scipy.stats.norm.ppf(uniform_returns['B'])

# Process returns for portfolio C (Normally-distributed)
scale = returns['A'].std()
uniform_returns['A'] = scipy.stats.norm.cdf(returns['A'], loc=0, scale=scale)
standard_normal_returns['A'] = scipy.stats.norm.ppf(uniform_returns['A'])

uniform_returns_df = pd.DataFrame(uniform_returns)
standard_normal_returns_df = pd.DataFrame(standard_normal_returns)
spearman_correlation, p_value = scipy.stats.spearmanr(standard_normal_returns_df, axis=0)

# If you specifically want a correlation matrix as output
correlation_matrix = pd.DataFrame(np.ones((2, 2)),
                                index=standard_normal_returns_df.columns, 
                                columns=standard_normal_returns_df.columns)
correlation_matrix.iloc[0, 1] = spearman_correlation
correlation_matrix.iloc[1, 0] = spearman_correlation

cov = correlation_matrix
mean = 0
mean_vector = [0,0]  # Assuming a mean vector of zeros for simplicity
simulated_normals = np.random.multivariate_normal(mean_vector, cov, size=n)

simU = pd.DataFrame(norm.cdf(simulated_normals), columns=returns.columns)


simulatedReturnsData = {}

std_dev_return = returns['A'].std()
simulatedReturnsData['A'] = norm.ppf(simU['A'], loc=0, scale=std_dev_return)

# Convert the simulated returns data into a DataFrame
simulatedReturnsData['B'] = t.ppf(simU['B'], deg_free['B'],loc=0 , scale=scale_t['B'])

simulatedReturns = pd.DataFrame(simulatedReturnsData)
#print(simulatedReturns)

# complete with model fitting
# Create a DataFrame for iterations
iterations = pd.DataFrame({'iteration': range(1, n)})

# Perform a cross join between Portfolio and iterations
values = pd.merge(portfolios.assign(key=1), iterations.assign(key=1), on='key').drop('key', axis=1)

# Calculate current value, simulated value, and PnL
values['currentValue'] = values['Holding'] * values['Starting Price']
values['simulatedValue'] = values.apply(lambda x: x['currentValue'] * (1.0 + simulatedReturns.loc[x['iteration'], x['Stock']]), axis=1)
values['pnl'] = values['simulatedValue'] - values['currentValue']
# values.to_csv('merged_values.csv', index=False)

gdf = values.groupby('iteration')
totalValues = gdf.agg({
    'currentValue': 'sum',
    'simulatedValue': 'sum',
    'pnl': 'sum'
}).reset_index()

print(totalValues)
#totalValues.to_csv('merged_values.csv', index=False)

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