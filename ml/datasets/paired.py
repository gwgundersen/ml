"""=============================================================================
A dataset with paired samples.
============================================================================="""

import numpy as np
from   ml.normalization import mean_center

# ------------------------------------------------------------------------------

Normal = np.random.multivariate_normal

# ------------------------------------------------------------------------------

def load(N=60, P=4, Q=3, exact=False, mean_centered=False):

    amean = np.zeros(P)
    acov  = np.eye(P)
    Xa    = Normal(mean=amean, cov=acov, size=N)

    if exact:
        Xb = Xa * 10
    else:
        bmean = np.ones(Q) * 2
        bcov  = np.eye(Q) * 1.5
        bcov[0, 1] = bcov[0, 2] = bcov[1, 2] = 10
        Xb    = Normal(mean=bmean, cov=bcov, size=N)

    if mean_centered:
        Xa = mean_center(Xa)
        Xb = mean_center(Xb)

    assert Xa.shape == (N, P)
    assert Xb.shape == (N, Q)
    return Xa, Xb
