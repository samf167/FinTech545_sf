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


# 1.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.covariance_skip_missing(A)
A_out.to_latex('1.1_out.txt', index=False)
print('1.1')
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
B.to_latex('1.1_expected.txt', index=False)
print("\n")

# 1.2
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.2.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.correlation_skip_missing(A)
A_out.to_latex('1.2_out.txt', index=False)
B.to_latex('1.2_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 1.3
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.3.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.covariance_pairwise(A)
A_out.to_latex('1.3_out.txt', index=False)
B.to_latex('1.3_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 1.4
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.4.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.correlation_pairwise(A)
A_out.to_latex('1.4_out.txt', index=False)
B.to_latex('1.4_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

#------------------------------------------------------------------------------------------
lam = 0.97

# 2.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_2.1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.ewCovar(A,0.97)
A_out = pd.DataFrame(A_out)
A_out.to_latex('2.1_out.txt', index=False)
B.to_latex('2.1_expected.txt', index=False)
print('2.1')
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 2.2
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_2.2.csv'
B = pd.read_csv(b)
A_out = fn.ewCorrelation(A,0.94)
A_out = pd.DataFrame(A_out)
A_out.to_latex('2.2_out.txt', index=False)
B.to_latex('2.2_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 2.3
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_2.3.csv'
B = pd.read_csv(b)
ew_covar_97 = fn.ewCovar(A, 0.97)
sd1 = np.sqrt(np.diag(ew_covar_97))
ew_covar_94 = fn.ewCovar(A, 0.94)
sd_inv = 1 / np.sqrt(np.diag(ew_covar_94))

diag_sd_inv = np.diag(sd_inv)
diag_sd1 = np.diag(sd1)

adjusted_covar = diag_sd1 @ diag_sd_inv @ ew_covar_94 @ diag_sd_inv @ diag_sd1

adjusted_covar_df = pd.DataFrame(adjusted_covar)
adjusted_covar_df.to_latex("2.3_out.txt", index=False)
B.to_latex('2.3_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(adjusted_covar_df))
print("Expected Norm:", np.linalg.norm(B))

# 3.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.3.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_3.1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.near_psd(A)
A_out = pd.DataFrame(A_out)
A_out.to_latex('3.1_out.txt', index=False)
B.to_latex('3.1_expected.txt', index=False)
print('3.1')
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 3.2
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.4.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_3.2.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.near_psd(A)
A_out = pd.DataFrame(A_out)
A_out.to_latex('3.2_out.txt', index=False)
B.to_latex('3.2_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 3.3
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.3.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_3.3.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.higham_nearest_psd(A.values)
A_out = pd.DataFrame(A_out)
A_out.to_latex('3.3_out.txt', index=False)
B.to_latex('3.3_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 3.4
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.4.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_3.4.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.higham_nearest_psd(A.values)
A_out = pd.DataFrame(A_out)
A_out.to_latex('3.4_out.txt', index=False)
B.to_latex('3.4_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 4.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_3.1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_4.1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
root = np.zeros_like(A)
A_out = fn.chol_psd(root, A.values)
A_out = pd.DataFrame(A_out)
print('4.1')
A_out.to_latex('4.1_out.txt', index=False)
B.to_latex('4.1_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")


# 5.1 
n = 100000
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test5_1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_5.1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
mean_vector = np.zeros(5) 
sim = np.random.multivariate_normal(mean_vector, A, size=n).T
A_out = np.cov(sim)
A_out = pd.DataFrame(A_out)
A_out.to_latex('5.1_out.txt', index=False)
B.to_latex('5.1_expected.txt', index=False)
print('5.1')
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))

print("\n")

# 5.2 
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test5_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_5.2.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
mean_vector = np.zeros(5) 
sim = np.random.multivariate_normal(mean_vector, A, size=n).T
A_out = np.cov(sim)
A_out = pd.DataFrame(A_out)
A_out.to_latex('5.2_out.txt', index=False)
B.to_latex('5.2_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 5.3
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test5_3.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_5.3.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
mean_vector = np.zeros(5) 
sim = np.random.multivariate_normal(mean_vector, fn.near_psd(A), size=n).T
A_out = np.cov(sim)
A_out = pd.DataFrame(A_out)
A_out.to_latex('5.3_out.txt', index=False)
B.to_latex('5.3_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 5.4 not working
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test5_3.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_5.4.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
mean_vector = np.zeros(5) 
sim = np.random.multivariate_normal(mean_vector, fn.higham_nearest_psd(A.values), size=n).T
A_out = np.cov(sim)
A_out = pd.DataFrame(A_out)
A_out.to_latex('5.4_out.txt', index=False)
B.to_latex('5.4_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 5.5 not workign
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test5_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_5.5.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
pca = fn.simulate_pca(A.values, n, 0.99)
cov_matrix = np.cov(pca, rowvar=False)
A_out = pd.DataFrame(cov_matrix)

A_out.to_latex('5.5_out.txt', index=False)
B.to_latex('5.5_expected.txt', index=False)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

#------------------------------------------------------------------------------------------
# 6.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test6.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test6_1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.arithmetic_calculate(A, "Date")
A_out = pd.DataFrame(A_out)
A_out.to_latex('6.1_out.txt', index=False)
B.to_latex('6.1_expected.txt', index=False)
A_out = A_out.set_index('Date').reset_index(drop=True)
B = B.set_index('Date').reset_index(drop=True)
print('6.1')
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

# 6.2
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test6.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test6_2.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.log_calculate(A, "Date")
A_out = pd.DataFrame(A_out)
A_out.to_latex('6.2_out.txt', index=False)
B.to_latex('6.2_expected.txt', index=False)
A_out = A_out.set_index('Date').reset_index(drop=True)
B = B.set_index('Date').reset_index(drop=True)
print("Realized Norm:", np.linalg.norm(A_out))
print("Expected Norm:", np.linalg.norm(B))
print("\n")

alpha = 0.05
n = 1000

# 7.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout7_1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.fit_norm(A)
A_out = pd.DataFrame(A_out)
A_out.to_latex('7.1_out.txt', index=False)
B.to_latex('7.1_expected.txt', index=False)
print("\n")

# 7.2
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout7_2.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.fit_t(A)
A_out = pd.DataFrame(A_out)
A_out.to_latex('7.2_out.txt', index=False)
B.to_latex('7.2_expected.txt', index=False)
print("\n")

# 7.3
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_3.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout7_3.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.fit_regression_t(A['y'], A[['x1', 'x2', 'x3']])
A_out = pd.DataFrame(A_out)
A_out.to_latex('7.3_out.txt', index=False)
B.to_latex('7.3_expected.txt', index=False)
print("\n")

# 8.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_1.csv'
B = pd.read_csv(b)
A = pd.read_csv(a)
A_out = fn.norm_var(A, alpha)
A_out = pd.DataFrame(A_out)
A_out.to_latex('8.1_out.txt', index=False)
B.to_latex('8.1_expected.txt', index=False)
print("\n")

# 8.2
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_2.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.t_var(A, alpha)

filename = "8.2_out.csv"
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([A_out])
B.to_latex('8.2_expected.txt', index=False)
print("\n")

# 8.3
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_3.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.hist_var(A, alpha, n)
filename = "8.3_out.csv"
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([A_out])
B.to_csv('8.3_expected.csv', index=False)
print("\n")

# 8.4
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_4.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.ES_norm(A, alpha)
filename = "8.4_out.csv"
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([A_out])
B.to_csv('8.4_expected.csv', index=False)
print("\n")

# 8.5
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_5.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.ES_t(A, alpha)
filename = "8.5_out.csv"
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([A_out])
B.to_csv('8.5_expected.csv', index=False)
print("\n")

# 8.6
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_6.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.ES_hist(A, alpha, n)
filename = "8.6_out.csv"
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([A_out])
B.to_csv('8.6_expected.csv', index=False)
print("\n")



# 9.1

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

