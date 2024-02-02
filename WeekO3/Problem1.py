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

filepath = "/Users/samfuller/Desktop/545/FinTech545_sf-1/WeekO3/DailyReturn.csv"
# Read the CSV data into a DataFrame
# Replace 'stock_returns.csv' with the path to your CSV file
df = pd.read_csv(filepath)

# Set the first column as the index if it represents dates
df.set_index(df.columns[0], inplace=True)

eigenvalue_list = [0.1,0.5,0.9,1]

# Define the span for exponential weighting
lambdal = 0.9
span = (2/lambdal) -1

# Calculate the weights and reorder so that time 1 weight comes first
weights = np.array([lambdal * (1 - lambdal)**i for i in range(len(df))][::-1])

# Ensure weights sum to 1
weights /= weights.sum()

# Center each column's data (subtract the mean)
centered_data = df - df.mean()

# Initialize an empty dataframe for covariance matrix
cov_matrix = pd.DataFrame(index=df.columns, columns=df.columns)


# Calculate the covariance matrix manually
for i in df.columns:
    for j in df.columns:
        cov_matrix.loc[i, j] = np.sum(weights * centered_data[i] * centered_data[j])

# Print the covariance matrix
print(cov_matrix)

pca = PCA(svd_solver='full')
pca.fit(cov_matrix)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative variance
plt.figure(figsize=(10, 7))
plt.plot(cumulative_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid(True)
plt.show()
