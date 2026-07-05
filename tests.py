from analytic_covariance_riemannian.utils import upper_herm, unupper_herm
import numpy as np

from analytic_covariance_riemannian.tangentspace import TangentSpaceHPD

# --- Test upper_herm and unupper_herm ---
def test_hermitian_transforms():
    print("\ntest_hermitian_transforms():")
    # Setup test parameters
    batch_size = 10
    n = 4

    # Generate random complex matrices
    A = np.random.randn(batch_size, n, n) + 1j * np.random.randn(batch_size, n, n)

    # Force them to be strictly Hermitian (X = X^H)
    X_orig = A + A.conj().transpose(0, 2, 1)

    print(f"Original shape: {X_orig.shape}")

    # 1. Apply forward transform
    Va_col = upper_herm(X_orig)
    print(f"Vectorized shape: {Va_col.shape}")

    # 2. Apply inverse transform
    X_reconstructed = unupper_herm(Va_col)
    print(f"Reconstructed shape: {X_reconstructed.shape}")

    # 3. Verify equality (accounting for floating point tolerances)
    is_match = np.allclose(X_orig, X_reconstructed)

    if is_match:
        print("\n SUCCESS: The reconstructed matrices perfectly match the originals.")
    else:
        print("\n FAILED: The reconstructed matrices do not match.")
        # Calculate maximum absolute error for debugging
        max_err = np.max(np.abs(X_orig - X_reconstructed))
        print(f"Max absolute error: {max_err}")


def test_upper_herm():
    print("\ntest_upper_herm():")
    # Test with a known Hermitian matrix
    X = np.array([[[1+0j, 2-1j], [2+1j, 3+0j]]])  # Shape (1, 2, 2)
    Va_col_expected = np.array([[1.0, np.sqrt(2.0)*2.0, 3.0, -np.sqrt(2.0)*1.0]])  # Real part of upper triangle (including diagonal) and imaginary part of strictly upper triangle
    Va_col = upper_herm(X)
    print("upper_herm output expected:", Va_col_expected)
    print("upper_herm output actual:", Va_col)
    print("All close?:", np.allclose(Va_col, Va_col_expected))

def test_unupper_herm():
    print("\ntest_unupper_herm():")
    # Test with a known vectorized Hermitian matrix
    Va_col = np.array([[1.0, np.sqrt(2.0)*2.0, 3.0, -np.sqrt(2.0)*1.0]])  # Shape (1, 4  )
    X_expected = np.array([[[1+0j, 2-1j], [2+1j, 3+0j]]])  # Shape (1, 2, 2)
    X = unupper_herm(Va_col)
    print("unupper_herm output expected:", X_expected)
    print("unupper_herm output actual:", X)
    print("All close?:", np.allclose(X, X_expected))


def test_TangentSpaceHPD():
    print("\ntest_TangentSpaceHPD():")
    # Generate random HPD matrices
    batch_size = 7
    n = 10
    A = np.random.randn(batch_size, n, n) + 1j * np.random.randn(batch_size, n, n)
    X = A@(A.conj().transpose(0, 2, 1))

    #print(X.shape)
    #print(X)

    ts_hpd = TangentSpaceHPD()
    Va_col = ts_hpd.fit_transform(X)
    X_reconstructed = ts_hpd.inverse_transform(Va_col)

    print("All close?:", np.allclose(X, X_reconstructed))


# Run the test
test_hermitian_transforms()
test_upper_herm()
test_unupper_herm()
test_TangentSpaceHPD()