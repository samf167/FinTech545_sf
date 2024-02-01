import numpy as np
from numpy.linalg import norm
import time

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


# PD matrix is if x^TAx geq 0 when x =/= 0 (exists nontrivial case)
# Eigen if (A - lambda(I))v = 0, Av = lambda(v)
# PSD is if x^TAx > 0 when x =/= 0 

# Higham

def _getAplus(A):
    eigval, eigvec = np.linalg.eigh(A)
    Q = np.matrix(eigvec)
    xdiag = np.diag(np.maximum(eigval, 0))
    return Q * xdiag * Q.T

def _getPs(A, W=None):
    W05 = np.sqrt(W) if W is not None else np.eye(A.shape[0])
    return np.linalg.inv(W05 @ A @ W05)

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def higham_nearest_psd(A, epsilon=0, max_iterations=100):
    n = A.shape[0]
    W = np.identity(n) 
    # Force A to be symmetric
    A = (A + A.T) / 2
    deltaS = 0
    Yk = A.copy()
    
    for k in range(max_iterations):
        Rk = Yk - deltaS
        Xk = _getAplus(Rk)
        deltaS = Xk - Rk
        Yk = _getPu(_getPs(Xk, W), W)
        # Stop condition
        normF = np.linalg.norm(Yk - A, 'fro')
        if normF < epsilon:
            break
            
    # Make the result symmetric
    Yk = (Yk + Yk.T) / 2
    
    # Ensure diagonals are 1
    Yk[np.diag_indices_from(Yk)] = 1
    
    return Yk

# Generate non-psd correlation matrix that is 500x500
n = 500

# Create an n x n matrix filled with 0.9
sigma = np.full((n, n), 0.9)

# Set the diagonal elements to 1.0
np.fill_diagonal(sigma, 1.0)

# Alter one pair of off-diagonal elements
sigma[0, 1] = 0.7357
sigma[1, 0] = 0.7357

# sigma is now a non-PSD correlation matrix
print("SIGMA",sigma)

start_time = time.time()
higams = higham_nearest_psd(sigma, epsilon=0, max_iterations=100)
higham_time = time.time() - start_time

near_fixed = near_psd(sigma, epsilon=0.0)

print(higams)
print(near_fixed)

# Prove correctness

# Compare using frombiaus norm 
highams_norm = norm(higams - sigma, 'fro')
near_norm = norm(near_fixed - sigma, 'fro')

