import numpy as np

def upper_blocks(X):
    """
    Concatenate the upper triangle (including diagonal) of the (1,1) 
    block and the upper triangle (excluding diagonal) of the (1,2) 
    block. To pereserve the geometry/inner product, apply sqrt(2) 
    scaling to (per block) off-diagonal elements.

    Returns real valued ndarray: shape (n_matrices, d), where
    d = half * half.
    """

    n = X.shape[-1]
    if X.shape[-2] != n or n % 2 != 0:
        raise ValueError("Matrix must be square with even size")
    half = n // 2

    # (1,1) block
    idx_11 = np.triu_indices(half)
    # Scaling: diagonal: 1, off-diagonal: sqrt(2)
    coeffs_11 = (np.sqrt(2) * np.triu(np.ones((half, half)), 1) + np.eye(half))[idx_11]
    T_11 = coeffs_11 * X[..., :half, :half][..., idx_11[0], idx_11[1]]

    # (1,2) block
    idx_12 = np.triu_indices(half,k=1)
    T_12 = np.sqrt(2) * X[..., :half, half:][..., idx_12[0], idx_12[1]]

    return np.concatenate([T_11, T_12], axis=-1)


def upper_herm(X):
    """
    Extracts the real part of the upper triangle (including diagonal)
    and the imaginary part of the strictly upper triangle (excluding 
    diagonal). To pereserve the geometry/inner product, apply sqrt(2) 
    scaling to off-diagonal elements

    Returns real valued ndarray: shape (n_matrices, d), where
    d = n*n.   
    """
    n = X.shape[-1]
    if X.shape[-2] != n:
        raise ValueError("Matrix must be square")
    
    X_real = X.real
    X_imag = X.imag

    # Real part: upper triangle including diagonal
    idx_real = np.triu_indices(n)
    coeffs_real = (np.sqrt(2) * np.triu(np.ones((n, n)), 1) + np.eye(n))[idx_real]
    T_real = coeffs_real * X_real[..., idx_real[0], idx_real[1]]

    # Imaginary part: strictly upper triangle (excluding diagonal)
    idx_imag = np.triu_indices(n, k=1)
    coeffs_imag = np.sqrt(2) * np.ones_like(idx_imag[0])
    T_imag = coeffs_imag * X_imag[..., idx_imag[0], idx_imag[1]]

    return np.concatenate([T_real, T_imag], axis=-1)