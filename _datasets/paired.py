"""============================================================================
A dataset with paired samples.
============================================================================"""

import numpy as np
from   _normalizers import mean_center

Normal = np.random.multivariate_normal


# -----------------------------------------------------------------------------

def load(N=60, P=4):
    amean = np.zeros(P)
    acov  = np.eye(P)
    Xa    = Normal(mean=amean, cov=acov, size=N)

    noise = np.random.normal(0, 1, size=(N, P)) * 0.00
    Xb = Xa * noise

    Xa = mean_center(Xa)
    Xb = mean_center(Xb)

    return Xa, Xb
