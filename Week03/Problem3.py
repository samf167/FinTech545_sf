import numpy as np
import scipy.stats as stats
import pandas as pd
import numpy as np
import statistics as st
import scipy
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
import seaborn
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from numpy.linalg import norm

filepath = "/Users/samfuller/Desktop/545/FinTech545_sf-1/WeekO3/DailyReturn.csv"
# Read the CSV data into a DataFrame
# Replace 'stock_returns.csv' with the path to your CSV file
df = pd.read_csv(filepath)
df_pearson = df

# Set the first column as the index if it represents dates
df.set_index(df.columns[0], inplace=True)
df_pearson.set_index(df.columns[0], inplace=True)

lambdal = 0.97
span = (2/lambdal) -1

# Calculate the weights and reorder so that time 1 weight comes first
weights = np.array([lambdal * (1 - lambdal)**i for i in range(len(df))][::-1])

# Ensure weights sum to 1
weights /= weights.sum()

# Center each column's data (subtract the mean)
centered_data = df - df.mean()

# Initialize an empty dataframe for covariance matrix
cov_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
var_vector = {col: 0 for col in df.columns}


# Calculate the covariance matrix manually
for i in df.columns:
    for j in df.columns:
        cov_matrix.loc[i, j] = np.sum(weights * centered_data[i] * centered_data[j])
    var_vector[i] = cov_matrix.loc[i, i]


# Use covariance results to calculate covariance div by variance 
for i in df.columns:
    for j in df.columns:
        corr_matrix.loc[i, j] = cov_matrix.loc[i, j] / (np.sqrt((cov_matrix.loc[i, i]) *(cov_matrix.loc[j, j])))


# Print the covariance matrix determined by exponential weighting
#print(cov_matrix)
#print(corr_matrix)
#print(var_vector)

# Calculate standard corr/variance
correlation_matrix = np.corrcoef(df_pearson, rowvar=False)
variance_vector = np.var(df_pearson, axis=0, ddof=1) # ddof=1 for sample varianc

#print(correlation_matrix)
#print(variance_vector)


# 1: exponential + exponential
cov_matrix_1 = pd.DataFrame(index=df.columns, columns=df.columns)
for i in df.columns:
    for j in df.columns:
        cov_matrix_1.loc[i, j] = corr_matrix.loc[i, j] * (np.sqrt((var_vector[i]) *(var_vector[j])))

# 2: exponential + pearson
cov_matrix_2 = pd.DataFrame(index=df.columns, columns=df.columns)
for i in df.columns:
    for j in df.columns:
        cov_matrix_2.loc[i, j] = corr_matrix.loc[i, j] * (np.sqrt((variance_vector[i]) *(variance_vector[j])))

# 3: pearson + pearson
cov_matrix_3 = pd.DataFrame(index=df.columns, columns=df.columns)
for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        cov_matrix_3.loc[df.columns[i], df.columns[j]] = correlation_matrix[i, j] * (np.sqrt((variance_vector[df.columns[i]]) *(variance_vector[df.columns[j]])))


# 4: pearson + exponential
cov_matrix_4 = pd.DataFrame(index=df.columns, columns=df.columns)
for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        cov_matrix_4.loc[df.columns[i], df.columns[j]] = correlation_matrix[i, j] * (np.sqrt((var_vector[df.columns[i]]) *(var_vector[df.columns[j]])))

print(cov_matrix_1)
print(cov_matrix_2)
print(cov_matrix_3)
print(cov_matrix_4)

matrix_list = [cov_matrix_1, cov_matrix_2, cov_matrix_3, cov_matrix_4]

# Assume 'cov_matrix' is your input covariance matrix.
num_draws = 25000
for i in matrix_list:
    # 1. Direct Simulation
    start_time = time.time()
    direct_samples = np.random.multivariate_normal(mean=np.zeros(i.shape[0]), cov=i, size=num_draws)
    direct_covariance = np.cov(direct_samples, rowvar=False)
    direct_time = time.time() - start_time
    direct_frobenius_norm = norm(direct_covariance - i, 'fro')

    # Function to perform PCA-based simulation with explained variance ratio
    def pca_simulation(cov_matrix, explained_variance, num_draws):
        pca = PCA()
        pca.fit(cov_matrix)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(cum_var >= explained_variance) + 1
        reduced_cov_matrix = np.diag(pca.explained_variance_[:num_components])
        eigenvalues_sqrt = np.sqrt(reduced_cov_matrix)
        eigenvectors = pca.components_[:num_components]
        z = np.random.normal(size=(num_draws, num_components))
        samples = z.dot(eigenvalues_sqrt).dot(eigenvectors) + pca.mean_
        return samples, time.time() - start_time

    # 2. PCA with 100% explained
    start_time = time.time()
    samples_100, time_100 = pca_simulation(i, 1.00, num_draws)
    covariance_100 = np.cov(samples_100, rowvar=False)
    frobenius_norm_100 = norm(covariance_100 - i, 'fro')

    # 3. PCA with 75% explained
    start_time = time.time()
    samples_75, time_75 = pca_simulation(i, 0.75, num_draws)
    covariance_75 = np.cov(samples_75, rowvar=False)
    frobenius_norm_75 = norm(covariance_75 - i, 'fro')

    # 4. PCA with 50% explained
    start_time = time.time()
    samples_50, time_50 = pca_simulation(i, 0.50, num_draws)
    covariance_50 = np.cov(samples_50, rowvar=False)
    frobenius_norm_50 = norm(covariance_50 - i, 'fro')
    print("Direct Covariance Matrix:", direct_covariance)
    print("100% PCA Covariance Matrix:",covariance_100)
    print("75% PCA Covariance Matrix:",covariance_75)
    print("50% PCA Covariance Matrix:", covariance_50)
    print(f"Direct Simulation: Frobenius Norm = {direct_frobenius_norm}, Time = {direct_time}")
    print(f"PCA 100%: Frobenius Norm = {frobenius_norm_100}, Time = {time_100}")
    print(f"PCA 75%: Frobenius Norm = {frobenius_norm_75}, Time = {time_75}")
    print(f"PCA 50%: Frobenius Norm = {frobenius_norm_50}, Time = {time_50}")