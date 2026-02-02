"""
Class for estimating analytic covariance matrices from multi-channel 
time-series data.

Note: Implementation largely follows pyriemann and scikit-learn 
conventions to ensure compatibility. Licensed under BSD-3-Clause (see 
LICENSE file).
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.signal import hilbert
from pyriemann.utils.covariance import covariances

class AnalyticCovariances(TransformerMixin, BaseEstimator):
    """Estimation of analytic covariance matrices.

    For each input, construct the analytic signal and estimate the
    sample analytic covariance matrix.
    
    Parameters
    ----------
    estimator : string, default="scm"
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    real_output : bool, default=False
        If True, the stacked HACOV (SPD) representation is used.
        If False, the complex-valued ACOV (HPD) representation is used.
    **kwds : dict
        Any further parameters are passed directly to the covariance 
        estimator.
    """

    def __init__(self, estimator="scm", real_output=False, **kwds):
        self.estimator = estimator
        self.real_output = real_output
        self.kwds = kwds
    

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, for compatibility with sklearn API.

        Returns
        -------
        self : AnalyticCovariances instance
            The AnalyticCovariances instance.
        """
        return self

    def transform(self, X):
        """Estimate covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        X_new : ndarray
            Covariance matrices. Shape and dtype depend on real_output:
            - If real_output is False (default): 
            shape (n_matrices, n_channels, 
                   n_channels), HPD matrices.
            - If real_output is True: 
            shape (n_matrices, 2*n_channels,
                   2*n_channels), stacked real/imag parts, SPD matrices.
        """

        X = np.asarray(X)
        mu = np.mean(X, axis=2, keepdims=True) # (n_samples, n_channels, 1)
        X_centered = X - mu

        # Compute analytic signal for all samples and channels
        Z = hilbert(X_centered, axis=-1) # (n_samples, n_channels, n_times)

        # Stack real and imaginary parts
        if self.real_output:
            Z = np.concatenate((Z.real, Z.imag), axis=1)
        
        covmats = covariances(Z, estimator=self.estimator, **self.kwds)
        
        return covmats
    
