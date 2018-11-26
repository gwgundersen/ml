"""=============================================================================
A dataset with paired samples, both with low rank covariance matrices.
============================================================================="""

import numpy as np

# ------------------------------------------------------------------------------

Normal = np.random.multivariate_normal
mm = np.matmul

# ------------------------------------------------------------------------------

def load(N=60, P=4, Q=3, k=10):

    amean = np.zeros(P)
    acov  = mm(np.random.randn(P, k), np.random.randn(k, P))
    Xa    = Normal(mean=amean, cov=acov, size=N)

    bmean = np.ones(Q) * 2
    bcov  = mm(np.random.randn(Q, k), np.random.randn(k, Q)) * 10
    Xb    = Normal(mean=bmean, cov=bcov, size=N)

    assert Xa.shape == (N, P)
    assert Xb.shape == (N, Q)
    return Xa, Xb
