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



def ledoit_wolf_complex(X, assume_centered=False):
    """
    Computes HPD Ledoit-Wolf shrinkage for a single (..., channels, samples) array.
    """
    # SCM
    p = X.shape[-2]
    n = X.shape[-1]

    if assume_centered:
        X_centered = X
    else:
        X_centered = X - np.mean(X, axis=-1, keepdims=True)

    # Transpose by swapping the last two axes
    S = (X_centered @ X_centered.conj().swapaxes(-2, -1)) / n

    mu = np.real(np.trace(S, axis1=-2, axis2=-1)) / p
    T = mu[..., None, None] * np.eye(p, dtype=complex)

    # Compute delta^2
    delta_squared = np.sum(np.abs(S - T)**2, axis=(-2, -1))

    # Compute beta^2
    time_point_squared_norms = np.sum(np.abs(X_centered)**2, axis=-2) #
    expected_norm_squared = np.mean(time_point_squared_norms**2, axis=-1)
    norm_S_squared = np.sum(np.abs(S)**2, axis=(-2, -1))
    beta_squared = (expected_norm_squared - norm_S_squared) / n

    # Compute shrinkage intensity and clip to [0, 1].
    shrinkage = np.zeros_like(delta_squared)
    mask = delta_squared != 0
    shrinkage[mask] = np.clip(beta_squared[mask] / delta_squared[mask], 0.0, 1.0)
    shrinkage[~mask] = 1.0

    # Expand 1D arrays to 3D so they can multiply against the matrices
    shrinkage_expanded = shrinkage[..., None, None]
    Sigma_shrunk = (1 - shrinkage_expanded) * S + shrinkage_expanded * T

    return Sigma_shrunk, shrinkage

def oas_complex(X, assume_centered=False):
    """
    Computes the HPD Oracle Approximating Shrinkage (OAS)
    natively for a complex (..., channels, samples) array.
    """
    # SCM
    p = X.shape[-2]
    n = X.shape[-1]

    if assume_centered:
        X_centered = X
    else:
        X_centered = X - np.mean(X, axis=-1, keepdims=True)

    # Transpose by swapping the last two axes
    S = (X_centered @ X_centered.conj().swapaxes(-2, -1)) / n

    # Target Matrix
    tr_S = np.real(np.trace(S, axis1=-2, axis2=-1))
    mu = tr_S / p
    T = mu[..., None, None] * np.eye(p, dtype=complex)

    # Compute U and V for the complex OAS formula for each matrix in the batch
    U = np.sum(np.abs(S)**2, axis=(-2, -1)) # ||S||_F^2
    V = tr_S**2 # tr(S)^2

    # The complex OAS shrinkage formula
    # From “Oracle Approximating Shrinkage Covariance Matrix Estimators for Complex Elliptical Distributions” by Elias Raninen.
    num = p * (p * V - U)
    den = (n*p - 1) * (p * U - V)

    # Compute shrinkage intensity and clip to [0, 1].
    shrinkage = np.zeros_like(den, dtype=float)
    mask = den != 0
    shrinkage[mask] = np.clip(num[mask] / den[mask], 0.0, 1.0)
    shrinkage[~mask] = 1.0

    # Expand 1D arrays to 3D so they can multiply against the matrices
    shrinkage_expanded = shrinkage[..., None, None]
    Sigma_shrunk = (1 - shrinkage_expanded) * S + shrinkage_expanded * T

    return Sigma_shrunk, shrinkage

def minimal_shrinkage_complex(X, assume_centered=False, shrinkage=1e-4):
    """
    Applies a fixed minimal shrinkage to ensure full rank matrices.
    Supports 2D (p, n) and batched (..., p, n) arrays.
    """
    p = X.shape[-2]
    n = X.shape[-1]

    if assume_centered:
        X_centered = X
    else:
        X_centered = X - np.mean(X, axis=-1, keepdims=True)

    # SCM
    S = (X_centered @ X_centered.conj().swapaxes(-2, -1)) / n

    # Target Matrix
    tr_S = np.real(np.trace(S, axis1=-2, axis2=-1))
    mu = tr_S / p
    T = mu[..., None, None] * np.eye(p, dtype=complex)

    # Apply the fixed shrinkage
    Sigma_shrunk = (1 - shrinkage) * S + shrinkage * T

    return Sigma_shrunk, shrinkage


def unupper_herm(Va_col):
    """
    Extracts the real part of the upper triangle (including diagonal)
    and the imaginary part of the strictly upper triangle (excluding
    diagonal). To pereserve the geometry/inner product, apply sqrt(2)
    scaling to off-diagonal elements

    Returns real valued ndarray: shape (n_matrices, d), where
    d = n*n.
    """
    n = np.sqrt(Va_col.shape[-1]).astype(int)
    if Va_col.shape[-1] != n * n:
        raise ValueError("Vector must correspond to a square matrix")

    # Indices for upper and strictly upper triangles
    idx_real = np.triu_indices(n)
    idx_imag = np.triu_indices(n, k=1)
    n_real = len(idx_real[0])

    Va_col_real = Va_col[...,:n_real]
    Va_col_imag = Va_col[...,n_real:]

    # make symmetric by filling in the lower triangle with the transpose of the upper triangle
    Va_real = np.zeros((Va_col.shape[0], n, n))
    Va_imag = np.zeros((Va_col.shape[0], n, n))

    # Reconstruct real part
    unscale_real = np.where(idx_real[0] == idx_real[1], 1.0, 1.0 / np.sqrt(2))
    Va_real[..., idx_real[0], idx_real[1]] = Va_col_real * unscale_real

    # Symmetrize to fill the strictly lower triangle (transpose of upper)
    Va_real[..., idx_imag[1], idx_imag[0]] = Va_real[..., idx_imag[0], idx_imag[1]]


    # Reconstruct the imaginary part
    Va_imag[..., idx_imag[0], idx_imag[1]] = Va_col_imag / np.sqrt(2)

    # Anti-symmetrize the strictly lower triangle to get a skew-symmetric imaginary part
    Va_imag[..., idx_imag[1], idx_imag[0]] = -Va_imag[..., idx_imag[0], idx_imag[1]]

    # Add real and imaginary parts together to form the full Hermitian matrices.
    Va = Va_real + 1j * Va_imag

    return Va
