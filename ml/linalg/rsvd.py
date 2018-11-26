"""=============================================================================
Randomized SVD.

See:

    Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions
    Halko, Martinsson, Tropp, SIAM 2011
============================================================================="""

import numpy as np
from   ml.normalization import normalize_columns

# ------------------------------------------------------------------------------

mm  = np.matmul

# ------------------------------------------------------------------------------

def find_Q(A, l):
    """Algorithm 4.1: Randomized range finder.

    Given an m × n matrix A, and an integer l, compute an m × l orthonormal
    matrix Q whose range approximates the range of A.
    """
    # Step 1. Draw an n × l Gaussian random matrix O.
    m, n = A.shape
    Ω = np.random.randn(n, l)

    # Step 2. Form the m × l matrix Y = AΩ.
    Y = mm(A, Ω)
    Y = normalize_columns(Y)

    # Step 3. Construct an m × l matrix Q whose columns are an orthonormal
    #         basis for the range of Y, e.g. using the QR factorization Y = QR.
    Q, R = np.linalg.qr(Y, mode='reduced')
    return Q

# ------------------------------------------------------------------------------

def rsvd(A, k):
    """Given an m × n matrix A, a target number k of singular vectors, and an
    exponent q (say q = 1 or q = 2), this procedure computes an approximate
    rank-2k factorization UΣVt, where U and V are orthonormal and Σ is
    nonnegative and diagonal.
    """
    # Stage A.
    Q = find_Q(A, 2 * k)

    # Stage B.
    B = mm(Q.T, A)
    S, Σ, Vt = np.linalg.svd(B)
    U = mm(Q, S)

    return U, Σ, Vt
