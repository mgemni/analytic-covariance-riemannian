import os
import math
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM

#from covariance import HPDCovarianceEstimator
from estimation import AnalyticCovariances

np.random.seed(42)

# Format the value with up to 2 decimal places
def formnr(value):
    formatted_value = f"{value:.2f}"
    return formatted_value

# ==========================================
# ===== Generate an artificial dataset =====
# ==========================================

def generate_data_for_class(n, ch, ts, f, mu, sigma, noise_std):
    X = np.zeros((n,ch,ts))
    param_dict = {}

    phis_relative = mu + sigma * np.random.randn(1, n) # Relative phase per signal class
    phis_absolute = np.concatenate((np.zeros((1, n)), phis_relative), axis=0) # Absolute phases
    #phis_absolute = np.random.uniform(0, 2 * math.pi, size=(1, n)) + np.concatenate((np.zeros((1, n)), phis_relative), axis=0) # If we want to randomize the absolute phase (does not make a difference to the result).
    phis = phis_absolute # Pick the phi's we want to use 

    freqs = f*np.ones((ch,1)) # Frequency for each data - constant for now.
    t = np.linspace(0, 1, ts)  # Time-index vector, from 0 to 1.

    # Save parameters, phis are used for plotting later
    param_dict["phis"] = phis
    param_dict["freqs"] = freqs
    param_dict["t"] = t

    # Create time-series
    for i in range(n):
        time_series_2d = np.cos(2 * np.pi * freqs * t + phis[:,i:i+1]) + noise_std * np.random.randn(ch, ts)
        X[i] = time_series_2d

    return X, param_dict


# =================
# ===== Setup =====
# =================

# Dataset parameters
sigma1 = 0.45
sigma2 = 0.90
noise_std = 0.10
n = 500 # nof data per class (dpc)
ts = 1000 # nof time-samples
f = 5 # frequency
ch = 2 # nof channels

# Evaluation parameters
nof_evals = 10
grid_points = 30
mu1s = np.linspace(-math.pi, math.pi, grid_points)
mu2s = np.linspace(-math.pi, math.pi, grid_points)

# Folder for saving data
save_folder = "results_ex2/dpc_{}_nofevals_{}_sigma1_{}_sigma2_{}_noise_std_{}_grid_{}".format(n,nof_evals,formnr(sigma1),formnr(sigma2),formnr(noise_std), grid_points)

# Pipelines
pipelines = {}
pipelines["COV+MDM"] = Pipeline([('cov', Covariances(estimator='cov')), ('mdm', MDM())])
pipelines["ACOV+MDM"] = Pipeline([('acov', AnalyticCovariances()), ('mdm', MDM())])

result_grid = {}
for key in pipelines.keys():
    result_grid[key] = np.zeros((mu1s.shape[0], mu2s.shape[0]))


# ============================================
# ===== Main loop, sweep mu1's and mu2's =====
# ============================================

for i, mu1 in enumerate(mu1s):
    print("Index mu1: {}, mu1: {}".format(i, mu1))
    for j, mu2 in enumerate(mu2s):
        print("Index mu2: {}, mu2: {}".format(j, mu2))

        scores = {}
        for name in pipelines.keys():
            scores[name] = []

        # Genereate data nof_evals times.
        for eval in range(nof_evals):
            
            # Generate data from 2 classes
            X_c1, param_dict_c1 = generate_data_for_class(n, ch, ts, f, mu1, sigma1, noise_std)
            X_c2, param_dict_c2 = generate_data_for_class(n, ch, ts, f, mu2, sigma2, noise_std)
            X = np.concatenate((X_c1, X_c2), axis=0)
            y = np.concatenate((np.ones(n), 2*np.ones(n))).astype(int)

            # Split data in train test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

            # Fit and predict with pipelines:
            for name, pipeline in pipelines.items():
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                scores[name].append(accuracy_score(y_test, y_pred))

        # Average the scores
        for name in pipelines.keys():
            result_grid[name][i,j] = np.mean(np.array(scores[name]))


# --- Save to .csv ---
if not os.path.exists(save_folder):
    os.makedirs("./" + save_folder)
    print(f"Folder '{save_folder}' created.")
else:
    print(f"Folder '{save_folder}' already exists.")

for key in pipelines.keys():
    print("Results for {}\n{}".format(key, result_grid[key]))
    df = pd.DataFrame(result_grid[key], index=mu1s.tolist(), columns=mu2s.tolist())
    df.to_csv("{}/data_{}.csv".format(save_folder,key), index=True, header=True) 
