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

# 1.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.covariance_skip_missing(A)
print("expected", B)
print("actual", A_out)
norm = norm(norm(A_out - B, 'fro'))
print("NORMNORMNORM", norm)
print("\n")

# 1.2
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.2.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.correlation_skip_missing(A)
print("expected", B)
print("actual", A_out)
print("\n")

# 1.3
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.3.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.covariance_pairwise(A)
print("expected", B)
print("actual", A_out)
print("\n")

# 1.4
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.4.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.correlation_pairwise(A)
print("expected", B)
print("actual", A_out)
print("\n")

#------------------------------------------------------------------------------------------
lam = 0.97

# 2.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_2.1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.ewCovar(A,lam)
print("expected", B)
print("actual", A_out)
print("\n")

# 2.2
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_2.2.csv'
B = pd.read_csv(b)
A_out = fn.ewCorrelation(A,lam)
print("expected", B)
print("actual", A_out)
print("\n")

# 2.3
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_2.3.csv'
B = pd.read_csv(b)
#A_out = fn.NOTDONE(A)

# 3.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_1.3.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_2.1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.near_psd(A)
print("expected", B)
print("actual", A_out)
print("\n")

'''
# 6.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test6.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_6.1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.arithmetic_calculate(A, "Date")
print("expected", B, "actual", A_out)

# 6.2
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test6.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout_6.2.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.log_calculate(A, "Date")
print("expected", B, "actual", A_out)
'''

#------------------------------------------------------------------------------------------
alpha = 0.05
n = 1000

# 7.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout7_1.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.fit_norm(A)
print("expected", B)
print("actual", A_out)
print("\n")

# 7.2
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout7_2.csv'
B = pd.read_csv(b)
A_out = fn.fit_log(A)
print("expected", B)
print("actual", A_out)
print("\n")

# 8.1
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_1.csv'
B = pd.read_csv(b)
A_out = fn.norm_var(A, alpha)
print("expected", B)
print("actual IGFRBDB", A_out)
print("\n")

# 8.2
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_2.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.t_var(A, alpha)
print("expected", B)
print("actual", A_out)
print("\n")

# 8.3
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_3.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.hist_var(A, alpha, n)
print("expected", B)
print("actual", A_out)
print("\n")

# 8.4
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_1.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_4.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.ES_norm(A, alpha)
print("expected", B)
print("actual", A_out)
print("\n")

# 8.5
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_5.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.ES_t(A, alpha)
print("expected", B)
print("actual", A_out)
print("\n")

# 8.6
a = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/test7_2.csv'
b = '/Users/samfuller/Desktop/545/FinTech545_sf-2/Library/testout8_6.csv'
A = pd.read_csv(a)
B = pd.read_csv(b)
A_out = fn.ES_hist(A, alpha, n)
print("expected", B)
print("actual 8.6", A_out)
print("\n")
