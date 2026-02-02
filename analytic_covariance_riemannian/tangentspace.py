"""
Extensions for pyriemann TangentSpace class.

This module defines custom subclasses of the standard TangentSpace class 
for implemention of vectorization strategies (Hermitian and block) used 
in the research paper [INSERT].

These classes override the transformation step to allow for 
custom logic defined in the local 'utils' module, while preserving 
compatibility with Scikit-Learn pipelines.

Note: Implementation largely follows pyRiemann and scikit-learn 
conventions to ensure compatibility. Licensed under BSD-3-Clause (see 
LICENSE file).
"""

# Imports
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.tangentspace import log_map_riemann as log_map

# Local import of utilities where the custom functions are implemented.
from analytic_covariance_riemannian import utils 

class TangentSpaceHPD(TangentSpace):
    """
    Tangentspace mapping using the vectorization strategy for 
    Hermitian matrices.
    
    This class inherits from pyriemann.tangentspace.TangentSpace and 
    overrides the vectorization step to use 'utils.upper_herm' instead 
    of the standard upper triangle extraction.
    """

    def transform(self, X):
        """
        Project matrices into the HPD tangent space and vectorize them.
        
        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of HPD matrices.

        Returns
        -------
        Va_col : ndarray, shape (n_matrices, n_channels * n_channels)
            Tangent space vectors (Hermitian matrices)
        """
        # Retrieve reference matrix calculated during .fit()
        Cref = self.reference_
        
        # Map to Tangent Space
        Va = log_map(X, Cref)#, metric=self.metric)
        
        # Vectorize: Use the custom Hermitian upper-triangle extraction.
        Va_col = utils.upper_herm(Va)
        
        return Va_col

    def fit_transform(self, X, y=None, sample_weight=None):
        """
        Fit and transform in a single function. Explicitly chains fit  
        and transform to ensure custom transform() logic is applied.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of HPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        Va_col : ndarray, shape (n_matrices, n_channels * n_channels)
            Vectorized matrices.
        """
        return self.fit(X, y=y, sample_weight=sample_weight).transform(X)
    

class TangentSpaceSub(TangentSpace):
    """
    Tangentspace mapping using the "block" vectorization strategy 
    motivated by the HACOV SPD matrix representation.

    This class inherits from pyriemann.tangentspace.TangentSpace and 
    overrides the vectorization step to use 'utils.upper_block' instead 
    of the standard upper triangle extraction.
    """

    def transform(self, X):
        """
        Project matrices into the tangent space and vectorize them.
        
        Parameters
        ----------
        X : ndarray, shape (n_matrices, 2 * n_channels, 2 * n_channels)
            Set of SPD matrices.

        Returns
        -------
        V_col : ndarray, shape (n_matrices, n_channels * n_channels)
            Tangent space vectors extracted using the block strategy.
        """
        # Retrieve reference matrix calculated during .fit()
        Cref = self.reference_
        
        # Map to Tangent Space
        V = log_map(X, Cref)#, metric=self.metric)
        
        # Vectorize: Use the custom Block upper-triangle extraction.
        V_col = utils.upper_blocks(V)
        
        return V_col

    def fit_transform(self, X, y=None, sample_weight=None):
        """
        Fit and transform in a single function. Explicitly chains fit  
        and transform to ensure custom transform() logic is applied.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, 2 * n_channels, 2 * n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        V_col : ndarray, shape (n_matrices, n_channels * n_channels)
            Vectorized matrices.
        """
        return self.fit(X, y=y, sample_weight=sample_weight).transform(X)


