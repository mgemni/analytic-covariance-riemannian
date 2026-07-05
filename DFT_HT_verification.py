import numpy as np
from scipy.signal import hilbert

def scipy_discrete_hilbert(x):
    """Computes the discrete Hilbert transform using scipy."""
    # scipy's hilbert returns the analytic signal: z = x + j*H(x)
    # We only want the imaginary part H(x)
    analytic_signal = hilbert(x)
    return np.imag(analytic_signal)

def discrete_hilbert(x):
    """Computes the standard discrete Hilbert transform."""
    N = len(x)
    X = np.fft.fft(x)
    H = np.zeros(N, dtype=complex)
    
    if N % 2 == 0:
        # Even length
        H[1:N//2] = -1j
        H[N//2+1:] = 1j
        # H[N//2] is strictly 0 to maintain odd symmetry and real output
    else:
        # Odd length
        H[1:(N+1)//2] = -1j
        H[(N+1)//2:] = 1j
        
    return np.real(np.fft.ifft(X * H))

def test_identities(N, label, enforce_nyquist_limit=False):
    # 1. Generate random zero-mean signals
    x = np.random.randn(N)
    y = np.random.randn(N)
    x -= np.mean(x)
    y -= np.mean(y)
    
    # 2. Enforce Nyquist band limitation if requested (only matters for even N)
    if enforce_nyquist_limit and N % 2 == 0:
        X, Y = np.fft.fft(x), np.fft.fft(y)
        X[N//2] = 0
        Y[N//2] = 0
        x = np.real(np.fft.ifft(X))
        y = np.real(np.fft.ifft(Y))
        
    # 3. Compute Hilbert transforms
    hx = discrete_hilbert(x)
    hy = discrete_hilbert(y)
    
    # 4. Compute covariances (expectations)
    C_xy = np.mean(x * y)
    C_hxhy = np.mean(hx * hy)
    
    C_xhy = np.mean(x * hy)
    C_hxy = np.mean(hx * y)
    
    # 5. Print results
    print("===== SELF IMPLEMENTATION =====")
    print(f"--- {label} (N={N}) ---")
    print(f"Identity 1: C_xy == C_hxhy")
    print(f"  C_xy   = {C_xy:.6f}")
    print(f"  C_hxhy = {C_hxhy:.6f}")
    print(f"  Difference: {abs(C_xy - C_hxhy):.2e}")
    
    print(f"\nIdentity 2: C_xhy == -C_hxy")
    print(f"   C_xhy = {C_xhy:.6f}")
    print(f"  -C_hxy = {-C_hxy:.6f}")
    print(f"  Difference: {abs(C_xhy - (-C_hxy)):.2e}\n")


    print("===== SCIPY =====")
    hx = scipy_discrete_hilbert(x)
    hy = scipy_discrete_hilbert(y)
    
    # 4. Compute covariances (expectations)
    C_xy = np.mean(x * y)
    C_hxhy = np.mean(hx * hy)
    
    C_xhy = np.mean(x * hy)
    C_hxy = np.mean(hx * y)
    
    # 5. Print results
    print(f"--- SCIPY: {label} (N={N}) ---")
    print(f"Identity 1: C_xy == C_hxhy")
    print(f"  C_xy   = {C_xy:.6f}")
    print(f"  C_hxhy = {C_hxhy:.6f}")
    print(f"  Difference: {abs(C_xy - C_hxhy):.2e}")
    
    print(f"\nIdentity 2: C_xhy == -C_hxy")
    print(f"   C_xhy = {C_xhy:.6f}")
    print(f"  -C_hxy = {-C_hxy:.6f}")
    print(f"  Difference: {abs(C_xhy - (-C_hxy)):.2e}\n")



def test_scipy_identities(N, label, enforce_nyquist_limit=False):
    # 1. Generate random zero-mean signals
    x = np.random.randn(N)
    y = np.random.randn(N)
    x -= np.mean(x)
    y -= np.mean(y)
    
    # 2. Enforce Nyquist band limitation if requested
    if enforce_nyquist_limit and N % 2 == 0:
        X, Y = np.fft.fft(x), np.fft.fft(y)
        X[N//2] = 0
        Y[N//2] = 0
        x = np.real(np.fft.ifft(X))
        y = np.real(np.fft.ifft(Y))
        
    # 3. Compute Hilbert transforms using Scipy
    hx = scipy_discrete_hilbert(x)
    hy = scipy_discrete_hilbert(y)
    
    # 4. Compute covariances (expectations)
    C_xy = np.mean(x * y)
    C_hxhy = np.mean(hx * hy)
    
    C_xhy = np.mean(x * hy)
    C_hxy = np.mean(hx * y)
    
    # 5. Print results
    print(f"--- SCIPY: {label} (N={N}) ---")
    print(f"Identity 1: C_xy == C_hxhy")
    print(f"  C_xy   = {C_xy:.6f}")
    print(f"  C_hxhy = {C_hxhy:.6f}")
    print(f"  Difference: {abs(C_xy - C_hxhy):.2e}")
    
    print(f"\nIdentity 2: C_xhy == -C_hxy")
    print(f"   C_xhy = {C_xhy:.6f}")
    print(f"  -C_hxy = {-C_hxy:.6f}")
    print(f"  Difference: {abs(C_xhy - (-C_hxy)):.2e}\n")


# Run the scenarios
"""
np.random.seed(42)


print("===== SCIPY =====")
# Case 1: Odd Length
test_scipy_identities(N=99, label="ODD CASE")

# Case 2: Even Length WITH Nyquist Band Limitation
test_scipy_identities(N=100, label="EVEN CASE (Nyquist Band-Limited)", enforce_nyquist_limit=True)

# Case 3: Even Length WITHOUT Nyquist Band Limitation
test_scipy_identities(N=100, label="EVEN CASE (NOT Band-Limited - FAILS)", enforce_nyquist_limit=False)
"""

#print("===== SELF IMPLEMENTATION =====")

# Run the scenarios
np.random.seed(42) # For reproducibility

# Case 1: Odd Length (Naturally immune to the Nyquist issue)
test_identities(N=99, label="ODD CASE")

# Case 2: Even Length WITH Nyquist Band Limitation (Identities hold)
test_identities(N=100, label="EVEN CASE (Nyquist Band-Limited)", enforce_nyquist_limit=True)

# Case 3: Even Length WITHOUT Nyquist Band Limitation (Identities FAIL)
test_identities(N=100, label="EVEN CASE (NOT Band-Limited - FAILS)", enforce_nyquist_limit=False)
