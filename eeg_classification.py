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
from analytic_covariance_riemannian.tangentspace import TangentSpaceSub, TangentSpaceHPD
from analytic_covariance_riemannian.estimation import AnalyticCovariances, AnalyticRegularizedCovariances


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

# ===== Non-regularized pipelines =====
# Standard RG pipelines
pipelines["COV+MDM"] = Pipeline([('cov', Covariances()), ('mdm', MDM())])
pipelines["COV+TSP+LR"] = Pipeline([('cov', Covariances()), ('tsp',TangentSpace()),('lr', LogisticRegression())])

# Pipelines using ACOV
pipelines["ACOV+MDM"] = Pipeline([('acov', AnalyticCovariances()), ('mdm', MDM())])
pipelines["ACOV+TSH+LR"] = Pipeline([('acov', AnalyticCovariances()), ('tsp_hpd', TangentSpaceHPD()),('lr', LogisticRegression())])

# Pipelines using HACOV
#pipelines["HACOV+MDM"] = Pipeline([('hacov', AnalyticCovariances(real_output=True)), ('mdm', MDM())])
#pipelines["HACOV+TSSUB+LR"] = Pipeline([('hacov', AnalyticCovariances(real_output=True)), ('tsp_sub', TangentSpaceSub()),('lr', LogisticRegression())])
# Tangent space pipeline using the non-efficient HACOV representation in the tangent space (not used in the paper).
#pipelines["HACOV+TS+LR"] = Pipeline([('hacov', AnalyticCovariances(real_output=True)), ('tsp', TangentSpace()),('lr', LogisticRegression())])


# ===== LW-regularized pipelines =====
# Standard RG pipelines
pipelines["COV(lw)+MDM"] = Pipeline([('cov_lw', Covariances(estimator='lwf')), ('mdm', MDM())])
pipelines["COV(lw)+TSP+LR"] = Pipeline([('cov_lw', Covariances(estimator='lwf')), ('tsp',TangentSpace()),('lr', LogisticRegression())])

# Pipelines using ACOV
pipelines["ACOV(lw)+MDM"] = Pipeline([('acov_lw', AnalyticCovariances(estimator='lwf')), ('mdm', MDM())])
pipelines["ACOV(lw)+TSH+LR"] = Pipeline([('acov_lw', AnalyticCovariances(estimator='lwf')), ('tsp_hpd', TangentSpaceHPD()),('lr', LogisticRegression())])

# Pipelines using HACOV
# pipelines["HACOV(lw)+MDM"] = Pipeline([('hacov_lw', AnalyticCovariances(estimator='lwf', real_output=True)), ('mdm', MDM())])
# pipelines["HACOV(lw)+TSSUB+LR"] = Pipeline([('hacov_lw', AnalyticCovariances(estimator='lwf', real_output=True)), ('tsp_sub', TangentSpaceSub()),('lr', LogisticRegression())])
# Tangent space pipeline using the non-efficient HACOV representation in the tangent space (not used in the paper).
#pipelines["HACOV(lw)+TS+LR"] = Pipeline([('hacov_lw', AnalyticCovariances(estimator='lwf', real_output=True)), ('tsp', TangentSpace()),('lr', LogisticRegression())])

# Complex LWF pipelines:
pipelines["ACOV(lwfc)+MDM"] = Pipeline([('alwf', AnalyticRegularizedCovariances(estimator='lwf')), ('mdm', MDM())])
pipelines["ACOV(lwfc)+TSH+LR"] = Pipeline([('alwf', AnalyticRegularizedCovariances(estimator='lwf')), ('tsp_hpd', TangentSpaceHPD()),('lr', LogisticRegression())])

# ===== OAS-regularized pipelines =====
# Standard RG pipelines
pipelines["COV(oas)+MDM"] = Pipeline([('cov_oas', Covariances(estimator='oas')), ('mdm', MDM())])
pipelines["COV(oas)+TSP+LR"] = Pipeline([('cov_oas', Covariances(estimator='oas')), ('tsp',TangentSpace()),('lr', LogisticRegression())])

# Pipelines using ACOV
pipelines["ACOV(oas)+MDM"] = Pipeline([('acov_oas', AnalyticCovariances(estimator='oas')), ('mdm', MDM())])
pipelines["ACOV(oas)+TSH+LR"] = Pipeline([('acov_oas', AnalyticCovariances(estimator='oas')), ('tsp_hpd', TangentSpaceHPD()),('lr', LogisticRegression())])

# Pipelines using HACOV
#pipelines["HACOV(oas)+MDM"] = Pipeline([('hacov_oas', AnalyticCovariances(estimator='oas', real_output=True)), ('mdm', MDM())])
#pipelines["HACOV(oas)+TSSUB+LR"] = Pipeline([('hacov_oas', AnalyticCovariances(estimator='oas', real_output=True)), ('tsp_sub', TangentSpaceSub()),('lr', LogisticRegression())])
# Tangent space pipeline using the non-efficient HACOV representation in the tangent space (not used in the paper).
#pipelines["HACOV(oas)+TS+LR"] = Pipeline([('hacov_oas', AnalyticCovariances(estimator='oas', real_output=True)), ('tsp', TangentSpace()),('lr', LogisticRegression())])

# Complex OAS pipelines:
pipelines["ACOV(oasc)+MDM"] = Pipeline([('aoas', AnalyticRegularizedCovariances(estimator='oas')), ('mdm', MDM())])
pipelines["ACOV(oasc)+TSH+LR"] = Pipeline([('aoas', AnalyticRegularizedCovariances(estimator='oas')), ('tsp_hpd', TangentSpaceHPD()),('lr', LogisticRegression())])




# ===========================================
# ===== Parameter grids for GridSearch ======
# ===========================================

# For each pipeline to test, define a parameter grid.
param_grid = {}

# ===== Non-regularized pipelines =====
# Standard RG pipelines
param_grid["COV+MDM"] = {}
param_grid["COV+TSP+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}
# Pipelines using ACOV
param_grid["ACOV+MDM"] = {}
param_grid["ACOV+TSH+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}
# Pipelines using HACOV
param_grid["HACOV+MDM"] = {}
param_grid["HACOV+TSSUB+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}
param_grid["HACOV+TS+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}


# ===== LW-regularized pipelines =====
# Standard RG pipelines
param_grid["COV(lw)+MDM"] = {}
param_grid["COV(lw)+TSP+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}
# Pipelines using ACOV
param_grid["ACOV(lw)+MDM"] = {}
param_grid["ACOV(lw)+TSH+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}
# Pipelines using HACOV
param_grid["HACOV(lw)+MDM"] = {}
param_grid["HACOV(lw)+TSSUB+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}
param_grid["HACOV(lw)+TS+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}
# Complex LWF pipelines:
param_grid["ACOV(lwfc)+MDM"] = {}
param_grid["ACOV(lwfc)+TSH+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}


# ===== OAS-regularized pipelines =====
# Standard RG pipelines
param_grid["COV(oas)+MDM"] = {}
param_grid["COV(oas)+TSP+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}

# Pipelines using ACOV
param_grid["ACOV(oas)+MDM"] = {}
param_grid["ACOV(oas)+TSH+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}
# Pipelines using HACOV
param_grid["HACOV(oas)+MDM"] = {}
param_grid["HACOV(oas)+TSSUB+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}
param_grid["HACOV(oas)+TS+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}
# Complex OAS pipelines:
param_grid["ACOV(oasc)+MDM"] = {}
param_grid["ACOV(oasc)+TSH+LR"] = {
    "lr__C": [0.2, 0.5, 1, 2, 5],
}


# ==============================================
# ===== Evaluate each pipeline using MOABB =====
# ==============================================

subject_list = [1,2,3,4,5,6,7,8,9]

for clf_name in pipelines.keys():

    dataset.subject_list = subject_list
    path =  "./results_eeg_lw_new_test_batch"

    evaluation = WithinSessionEvaluation(paradigm=paradigm,
                                     datasets=dataset,
                                     overwrite=True,
                                     random_state=42,
                                     hdf5_path=path,
                                     save_model=True,
                                     n_jobs=-4,)
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
