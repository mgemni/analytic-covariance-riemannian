import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from moabb.paradigms import MotorImagery
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM


# Imports from custom files in this repo
from tangentspace import TangentSpaceSub, TangentSpaceHPD
from estimation import AnalyticCovariances


# Initialize parameter for the Band Pass filter
fmin = 8
fmax = 35
tmin = 0
tmax = None

# Load the dataset
dataset = BNCI2014001()
events = ["right_hand", "left_hand", "feet", "tongue"]
paradigm = MotorImagery(events=events, n_classes=len(events), fmin=fmin, fmax=fmax, tmax=tmax)


# =================================
# ===== Classifiers/Pipelines =====
# =================================

# Define the pipelines to test in a dictionary.
pipelines = {}

# Standard RG pipelines
pipelines["COV+MDM"] = Pipeline([('cov', Covariances()), ('mdm', MDM())])
pipelines["COV+TSP+LR"] = Pipeline([('cov', Covariances()), ('tsp',TangentSpace()),('lr', LogisticRegression())])

# MDM pipelines using ACOV and HACOV
pipelines["ACOV+MDM"] = Pipeline([('acov', AnalyticCovariances()), ('mdm', MDM())])
#pipelines["HACOV+MDM"] = Pipeline([('hacov', AnalyticCovariances(real_output=True)), ('mdm', MDM())])

# Tangent space based pipelines using ACOV and HACOV
pipelines["ACOV+TSH+LR"] = Pipeline([('acov', AnalyticCovariances()), ('tsp_hpd', TangentSpaceHPD()),('lr', LogisticRegression())])
#pipelines["HACOV+TSSUB+LR"] = Pipeline([('hacov', AnalyticCovariances(real_output=True)), ('tsp_sub', TangentSpaceSub()),('lr', LogisticRegression())])

# Tangent space pipeline using the non-compressed HACOV repersentation in the tangent space (not used in the paper).
#pipelines["HACOV+TS+LR"] = Pipeline([('hacov', AnalyticCovariances(real_output=True)), ('tsp', TangentSpace()),('lr', LogisticRegression())])


# ===========================================
# ===== Parameter grids for GridSearch ======
# ===========================================

# For each pipeline to test, define a parameter grid.
param_grid = {}

# Standard RG pipelines
param_grid["COV+MDM"] = { #Cov+MDM
    'cov__estimator': ["cov"],
}
param_grid["COV+TSP+LR"] = {
    'cov__estimator': ["cov"],
    "lr__C": [0.2, 0.5, 1, 2, 5],  # 5 values, log-ish spacing
    #"lr__C": [0.2, 0.5, 1, 2, 5, 10],  # 5 values, log-ish spacing
}

# MDM pipelines using ACOV and HACOV
param_grid["ACOV+MDM"] = {
    'acov__estimator': ["cov"],

}
param_grid["HACOV+MDM"] = {
    'hacov__estimator': ["cov"],
}

# Tangent space based pipelines using ACOV and HACOV
param_grid["ACOV+TSH+LR"] = {
    'acov__estimator': ["cov"],
    "lr__C": [0.2, 0.5, 1, 2, 5],  # 5 values, log-ish spacing
    #"lr__C": [0.2, 0.5, 1, 2, 5, 10],  # 5 values, log-ish spacing
}
param_grid["HACOV+TSSUB+LR"] = {
    'hacov__estimator': ["cov"],
    "lr__C": [0.2, 0.5, 1, 2, 5],  # 5 values, log-ish spacing
    #"lr__C": [0.2, 0.5, 1, 2, 5, 10],  # 5 values, log-ish spacing
}

# Tangent space pipeline using the non-compressed HACOV repersentation in the tangent space.
param_grid["HACOV+TS+LR"] = {
    'hacov__estimator': ["cov"],
    "lr__C": [0.2, 0.5, 1, 2, 5],  # 5 values, log-ish spacing
    #"lr__C": [0.2, 0.5, 1, 2, 5, 10],  # 5 values, log-ish spacing
}


# ==============================================
# ===== Evaluate each pipeline using MOABB =====
# ==============================================

subject_list = [1,2,3,4,5,6,7,8,9]

for clf_name in pipelines.keys():

    dataset.subject_list = subject_list
    path =  "./results_eeg"

    evaluation = WithinSessionEvaluation(paradigm=paradigm,
                                     datasets=dataset,
                                     overwrite=True,
                                     random_state=42,
                                     hdf5_path=path,
                                     save_model=True,
                                     n_jobs=-1,)
    # Run the evaluation
    result = evaluation.process({clf_name: pipelines[clf_name]}, {clf_name: param_grid[clf_name]})
    result.to_csv(os.path.join(path,"results_{}.csv".format(clf_name)))

    # Print the results
    print(result)
    print(result["score"].mean())


# -- Calculate and print average results per pipeline --
for clf_name in pipelines.keys():
    results = pd.read_csv(os.path.join(path,"results_{}.csv".format(clf_name)))
    print(f"Average score for {clf_name}: {results['score'].mean()}")
