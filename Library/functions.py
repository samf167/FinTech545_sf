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

# --------------------------Missing Data--------------------------------#

# 1.1 Cov missing
def covariance_skip_missing(dataframe):
    # Drop rows with any missing values
    clean_df = dataframe.dropna()
    # Calculate covariance matrix
    return clean_df.cov()

# 1.2 Corr missing
def correlation_skip_missing(dataframe):
    # Drop rows with any missing values
    clean_df = dataframe.dropna()
    # Calculate correlation matrix
    return clean_df.corr()

# 1.3 Cov missing (pairwise) 
def covariance_pairwise(dataframe):
    # Calculate covariance matrix with pairwise deletion
    return dataframe.cov(min_periods=1)

# 1.4 Corr missing (pairwise)
def correlation_pairwise(dataframe):
    # Calculate correlation matrix with pairwise deletion
    return dataframe.corr(min_periods=1)

# -------------------------Cov/Corr Calcs--------------------------------#

# 2.1 EW Covariance
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

# 2.2 EW Correlation
def ewCorrelation(X, lam):
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
    
    # Standard deviations
    std_devs = np.sqrt(np.diag(cov_matrix))
    
    # Create the correlation matrix
    corr_matrix = np.divide(cov_matrix, std_devs[:, None])
    corr_matrix = np.divide(corr_matrix, std_devs[None, :])
    
    return corr_matrix

# 2.3 

# 3.1 Near covar
def near_psd(a, epsilon=0.0):
    n = a.shape[0]
    
    # Copy the input matrix to output
    out = np.copy(a)
    
    # Calculate the correlation matrix if we got a covariance matrix
    if not np.allclose(np.diag(out), np.ones(n)):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    
    # Eigenvalue decomposition, update the eigenvalues and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = np.diag(1.0 / np.sqrt(np.sum(vecs**2 * vals, axis=1)))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T
    
    # Add back the variance
    if 'invSD' in locals():
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out


# 2.Higham
def _getAplus(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def _getPs(A, W=None):
    if W is None:
        W = np.eye(len(A))
    W05 = np.sqrt(W)
    iW = np.linalg.inv(W05)
    return iW @ _getAplus(W05 @ A @ W05) @ iW

def _getPu(A, W=None):
    Aret = A.copy()
    np.fill_diagonal(Aret, 1)
    return Aret

def wgtNorm(A, W=None):
    if W is None:
        W = np.eye(len(A))
    W05 = np.sqrt(W)
    W05 = W05 @ A @ W05
    return np.sum(W05 * W05)

def higham_nearest_psd(pc, W=None, epsilon=1e-9, max_iter=100, tol=1e-9):
    n = pc.shape[0]
    if W is None:
        W = np.eye(n)
    
    delta_s = 0
    Yk = pc.copy()
    norml = np.finfo(np.float64).max
    i = 1
    
    while i <= max_iter:
        Rk = Yk - delta_s
        Xk = _getPs(Rk, W)
        delta_s = Xk - Rk
        Yk = _getPu(Xk, W)
        norm = wgtNorm(Yk - pc, W)
        min_eig_val = np.min(np.real(np.linalg.eigvals(Yk)))
        
        if abs(norm - norml) < tol and min_eig_val > -epsilon:
            break
        
        norml = norm
        i += 1
    return Yk

# 4.1 Cholesky psd
def chol_psd(root, a):
    n = a.shape[0]
    # Initialize the root matrix with 0 values
    root[:] = 0.0

    # Loop over columns
    for j in range(n):
        s = 0.0
        # If we are not on the first column, calculate the dot product of the preceding row values.
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        # Diagonal Element
        temp = a[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        # Check for the 0 eigenvalue. Just set the column to 0 if we have one.
        if root[j, j] == 0.0:
            root[j, (j + 1):n] = 0.0
        else:
            # Update off-diagonal rows of the column
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir
    return s


# 6.1 Calc arithmetic returns

def arithmetic_calculate(prices, date_column="date"):
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame: {prices.columns}")

    # Exclude the date column from the calculation
    vars = prices.columns.difference([date_column])

    # Extract the price data excluding the date column
    p = prices[vars].values

    # Calculate simple returns
    returns = p[1:] / p[:-1] - 1

    # Add the date column to the returns DataFrame
    returns_df = pd.DataFrame(data=returns, columns=vars)
    returns_df[date_column] = prices[date_column].iloc[1:].values

    # Reorder columns to have the date column at the beginning
    cols = [date_column] + [col for col in returns_df.columns if col != date_column]
    returns_df = returns_df[cols]

    return returns_df


# 6.2 Calc log returns
def log_calculate(prices, date_column="date"):
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame: {prices.columns}")

    # Exclude the date column from the calculation
    vars = prices.columns.difference([date_column])

    # Extract the price data excluding the date column
    p = prices[vars].values

    # Calculate log returns
    returns = np.log(p[1:] / p[:-1])

    # Add the date column to the returns DataFrame
    returns_df = pd.DataFrame(data=returns, columns=vars)
    returns_df[date_column] = prices[date_column].iloc[1:].values

    # Reorder columns to have the date column at the beginning
    cols = [date_column] + [col for col in returns_df.columns if col != date_column]
    returns_df = returns_df[cols]

    return returns_df

# 7.1 Fit normal
def fit_norm(X):
    return(scipy.stats.norm.fit(X))

# 7.2 Fit T Dist
def fit_log(X):
    return(scipy.stats.t.fit(X))

# 8.1 Var from normal
def norm_var(X, alpha):
    sdev = X.std() # get stdev for sample set
    z_score = norm.ppf(alpha) # get inv cdf value
    VaR_norm = -((z_score * sdev)) - X.mean() # get VaR absolute
    return VaR_norm

# 8.2 Var from T
def t_var(X, alpha):
    params = scipy.stats.t.fit(X, method='MLE') # Fit T distribution
    alpha_return = scipy.stats.t.ppf(alpha, params[0], loc = params[1], scale =params[2]) # Save Df
    VaR_mle_t = (-alpha_return) # calc VaR
    return VaR_mle_t

# 8.3 VaR from Simulation
def hist_var(X, alpha, n):
    samples_list = [] # initialize sample list

    for i in range(n):
        new_samples = X.sample(n=10, replace=True) 
        samples_list.append(new_samples)

        sample_draws = pd.concat(samples_list, ignore_index=True)

        # Get alpha percentile for VaR
        VaR_historic = -np.percentile(sample_draws, alpha*100)

    return VaR_historic

# 8.4 ES from Norm
def ES_norm(X, alpha):
    mu, sdev = scipy.stats.norm.fit(X)
    z_score = norm.ppf(alpha) # get inv cdf value
    VaR_norm = -((z_score * sdev)) # get VaR

    phi_inv_alpha = norm.ppf(alpha)
    
    f_phi_inv_alpha = norm.pdf(phi_inv_alpha)
    
    ES = -mu + sdev * (f_phi_inv_alpha / alpha)
    return ES

# 8.5 ES from T 
def ES_t(X, alpha):
    params = scipy.stats.t.fit(X, method='MLE') # Fit T distribution
    alpha_return = scipy.stats.t.ppf(alpha, params[0], loc = params[1], scale =params[2]) # Save Df
    VaR_mle_t = (-alpha_return) # calc VaR
    integral, error = quad(lambda x: x * t.pdf(x, params[0], loc = params[1], scale =params[2]), -np.inf, -VaR_mle_t)
    # Calculate ES
    ES = -1/alpha * integral

    return ES

# 8.6 ES from Simulation
def ES_hist(X, alpha, n):
    samples_list = [] # initialize sample list

    # Bootstrap 10000 values from the historical x_df to approx distribution
    for i in range(n):
        new_samples = X.sample(n=10, replace=True) 
        samples_list.append(new_samples)

    sample_draws = pd.concat(samples_list, ignore_index=True)

    # Get alpha percentile for VaR
    VaR_historic = -np.percentile(sample_draws, alpha*100)

    # Calculate Es by brute force
    total = 0
    count = 0
    val = 0
    for i in range(n*(10)):
        val = sample_draws.loc[i, 'x1']
        if val < -VaR_historic:
            count+=1
            total= total + val

    ES_historic = -total/count
    return ES_historic