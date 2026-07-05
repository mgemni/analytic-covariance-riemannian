import numpy as np
from moabb.datasets import BNCI2014_001 # Replace with your dataset
from moabb.paradigms import MotorImagery # Replace with your paradigm
from sklearn.covariance import ledoit_wolf, oas

# 1. Initialize
dataset = BNCI2014_001()
paradigm = MotorImagery()
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[1]) 

# 2. Storage arrays
cond_raw, cond_lw, cond_oas = [], [], []
min_eig_raw, min_eig_lw, min_eig_oas = [], [], []
alphas_lw, alphas_oas = [], []

# 3. Loop over every trial
for trial in X:
    # A. Raw Sample Covariance
    C_raw = np.cov(trial)
    
    # B. Regularized Covariances (Note the trial.T)
    C_lw, alpha_lw = ledoit_wolf(trial.T)
    C_oas, alpha_oas = oas(trial.T)
    
    alphas_lw.append(alpha_lw)
    alphas_oas.append(alpha_oas)
    
    # C. Condition Numbers
    cond_raw.append(np.linalg.cond(C_raw))
    cond_lw.append(np.linalg.cond(C_lw))
    cond_oas.append(np.linalg.cond(C_oas))
    
    # D. Smallest Eigenvalues 
    min_eig_raw.append(np.min(np.linalg.eigvalsh(C_raw)))
    min_eig_lw.append(np.min(np.linalg.eigvalsh(C_lw)))
    min_eig_oas.append(np.min(np.linalg.eigvalsh(C_oas)))

# 4. Summary Statistics
print("=== CONDITION NUMBERS (κ) ===")
print(f"Raw: Median = {np.median(cond_raw):.2e}  (Mean ± SD = {np.mean(cond_raw):.2e} ± {np.std(cond_raw):.2e})")
print(f"LW:  Median = {np.median(cond_lw):.2f}    (Mean ± SD = {np.mean(cond_lw):.2f} ± {np.std(cond_lw):.2f})")
print(f"OAS: Median = {np.median(cond_oas):.2f}    (Mean ± SD = {np.mean(cond_oas):.2f} ± {np.std(cond_oas):.2f})")

print("\n=== SMALLEST EIGENVALUES (λ_min) ===")
print(f"Raw: Mean ± SD = {np.mean(min_eig_raw):.4f} ± {np.std(min_eig_raw):.4f}")
print(f"LW:  Mean ± SD = {np.mean(min_eig_lw):.4f} ± {np.std(min_eig_lw):.4f}")
print(f"OAS: Mean ± SD = {np.mean(min_eig_oas):.4f} ± {np.std(min_eig_oas):.4f}")

print("\n=== SHRINKAGE INTENSITY (α) ===")
print(f"LW:  Mean ± SD = {np.mean(alphas_lw):.4f} ± {np.std(alphas_lw):.4f}")
print(f"OAS: Mean ± SD = {np.mean(alphas_oas):.4f} ± {np.std(alphas_oas):.4f}")