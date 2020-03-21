"""=============================================================================
A dataset with paired samples, both with low rank covariance matrices.
============================================================================="""

import numpy as np

# ------------------------------------------------------------------------------

Normal = np.random.multivariate_normal
mm     = np.matmul
rank   = np.linalg.matrix_rank

# ------------------------------------------------------------------------------

def load(N=200, P=100, Q=100, k=10):
    Xa = generate_data(N, P, k, 1)
    Xb = generate_data(N, Q, k, 2)
    return Xa, Xb

# ------------------------------------------------------------------------------

def generate_data(N, P, k, scale):
    mu  = np.zeros(P)
    A   = mm(np.random.randn(P, k), np.random.randn(k, P)) * scale
    AA  = mm(A, A.T)
    eigvals = np.linalg.eigvals(AA)
    is_pos  = eigvals > 0
    is_zero = np.isclose(eigvals, 0)
    assert (is_pos | is_zero).all()

    # Wikipedia:
    #
    #     "For any matrix A, the matrix A*A is positive semidefinite, and
    #      rank(A) = rank(A*A)"
    #
    assert rank(A)  == k
    assert rank(AA) == rank(A)
    X = Normal(mean=mu, cov=AA, size=N)
    assert X.shape == (N, P)

    return X

# ------------------------------------------------------------------------------

def is_psd(A):
    return np.all(np.linalg.eigvals(A) >= 0)
