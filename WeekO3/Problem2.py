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

# Generate non-psd correlation matrix that is 500x500
n = 2000

# Create an n x n matrix filled with 0.9
sigma = np.full((n, n), 0.9)

# Set the diagonal elements to 1.0
np.fill_diagonal(sigma, 1.0)

# Alter one pair of off-diagonal elements
sigma[0, 1] = 0.7357
sigma[1, 0] = 0.7357

# sigma is now a non-PSD correlation matrix
#print("SIGMA",sigma)

start_time = time.time()
highams = higham_nearest_psd(sigma)
higham_time = time.time() - start_time

near_time = time.time()
near_fixed = near_psd(sigma, epsilon=0.0)
near_time = time.time() - start_time

highams_norm = norm(highams - sigma, 'fro')
near_norm = norm(near_fixed - sigma, 'fro')

print("Higham Output:", highams)
print("Higham Time:", higham_time)
print("Higham Norm:", highams_norm)

print("Near Output:", near_fixed)
print("Near Time:", near_time)
print("Near Norm:", near_norm)

# Prove correctness

# Compare using frombiaus norm 


