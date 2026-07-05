import numpy as np
from moabb.datasets import BNCI2014_001 # Replace with your dataset
from moabb.paradigms import MotorImagery # Replace with your paradigm
from sklearn.covariance import ledoit_wolf, oas

from analytic_covariance_riemannian.estimation import AnalyticCovariances

# 1. Initialize
dataset = BNCI2014_001()
paradigm = MotorImagery()
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[1]) 


print("X_shape:", X.shape)


acov = AnalyticCovariances(estimator="oas", real_output=False) # Try with "lw" and real_output=True as well
covmats = acov.transform(X)

print("Covmats shape:", covmats.shape)
print("Covmats dtype:", covmats.dtype)

print(covmats[0,:5,:5])
