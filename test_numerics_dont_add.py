import numpy as np
from moabb.datasets import BNCI2014_001 # Replace with your dataset
from moabb.paradigms import MotorImagery # Replace with your paradigm
from sklearn.covariance import ledoit_wolf, oas

from scipy.signal import hilbert
from pyriemann.estimation import Covariances

# 1. Initialize
dataset = BNCI2014_001()
paradigm = MotorImagery()
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[1]) 



Z = hilbert(X, axis=-1) # (n_samples, n_channels, n_times)


cov_est = Covariances()

C_r = cov_est.transform(Z.real)
C_i = cov_est.transform(Z.imag)

print(np.allclose(C_r, C_i))



Z_stacked = np.concatenate((Z.real, Z.imag), axis=1)

C_stacked = cov_est.transform(Z_stacked)

print("C_stacked shape:", C_stacked.shape)
print("C_stacked dtype:", C_stacked.dtype)

print(np.allclose(C_stacked[:,:X.shape[1],:X.shape[1]], C_r))
print(np.allclose(C_stacked[:,X.shape[1]:,X.shape[1]:], C_r))

print(np.allclose(C_stacked[:,X.shape[1]:,:X.shape[1]], -C_stacked[:,:X.shape[1],X.shape[1]:]))
