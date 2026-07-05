import numpy as np
from moabb.datasets import BNCI2014_001 # Replace with your dataset
from moabb.paradigms import MotorImagery # Replace with your paradigm
from sklearn.covariance import ledoit_wolf, oas
from scipy.signal import hilbert

from analytic_covariance_riemannian.estimation import AnalyticCovariances
from analytic_covariance_riemannian.utils import ledoit_wolf_complex as lw_complex

from analytic_covariance_riemannian.estimation import AnalyticCovariances, AnalyticLWF

# 1. Initialize
dataset = BNCI2014_001()
paradigm = MotorImagery()
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[1]) 



def compute_beta_squared_looped(X_centered, S):
    """
    Explicit, slow, looped calculation of beta_squared for complex data.
    """
    n, p = X_centered.shape
    beta_sum = 0.0
    
    for k in range(n):
        # Extract the k-th sample and reshape to a column vector (p, 1)
        x_k = X_centered[k, :].reshape(p, 1)
        
        # Calculate the outer product x_k * x_k^H (results in p x p matrix)
        x_k_outer = x_k @ x_k.conj().T
        
        # Calculate squared Frobenius norm of the difference
        diff = x_k_outer - S
        squared_frobenius = np.sum(np.abs(diff)**2)
        
        beta_sum += squared_frobenius
        
    return beta_sum / (n**2)


mu = np.mean(np.asarray(X), axis=2, keepdims=True) # (n_samples, n_channels, 1)
X_centered = X - mu

# Compute analytic signal for all samples and channels
Z = hilbert(X_centered, axis=-1) # (n_samples, n_channels, n_times)

X_stacked = np.concatenate((Z.real, Z.imag), axis=1)

i = 0
X_i = X_stacked[i]
Z_i = Z[i]

acov_est = AnalyticCovariances()
C_acovs = acov_est.transform(X)

alwf_est = AnalyticLWF()
C_alwfs = alwf_est.transform(X)



C_lwc, alpha_lwc = lw_complex(Z_i)
C_lw, alpha_lw = ledoit_wolf(X_i.T) 

print("LWC alpha:", alpha_lwc)
print("LW alpha:", alpha_lw)

print("LWC cov shape:", C_lwc.shape)
print("LW cov shape:", C_lw.shape)
print(C_lwc[:5,:5], "\n")
print(C_lw[:5,:5], "\n")
print(C_lw[22:27,:5], "\n")

print(C_acovs[i,:5,:5], "\n")
print(C_alwfs[i,:5,:5], "\n")

"""
# ==========================================
# Comparison Test
# ==========================================
np.random.seed(42)
n, p = 100, 10
# Generate complex data
X = np.random.randn(n, p) + 1j * np.random.randn(n, p)

# Center and calculate S
X_centered = X - np.mean(X, axis=0)
S = (X_centered.conj().T @ X_centered) / n

# 1. Efficient Vectorized Version (from the original code)
X_squared_norms = np.sum(np.abs(X_centered)**2, axis=1) 
expected_norm_squared = np.mean(X_squared_norms**2)
norm_S_squared = np.sum(np.abs(S)**2)
beta_squared_vectorized = (expected_norm_squared - norm_S_squared) / n

# 2. Explicit Looped Version
beta_squared_looped = compute_beta_squared_looped(X_centered, S)

print(f"Vectorized beta^2: {beta_squared_vectorized:.10f}")
print(f"Looped beta^2:     {beta_squared_looped:.10f}")
print(f"Difference:        {abs(beta_squared_vectorized - beta_squared_looped):.2e}"
"""