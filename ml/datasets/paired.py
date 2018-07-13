"""=============================================================================
A dataset with paired samples.
============================================================================="""

import numpy as np

# ------------------------------------------------------------------------------

def load(exact=False):
    N, P, Q = 60, 4, 3

    a1 = np.random.normal(loc=0, scale=1, size=(N,))
    a2 = np.random.normal(loc=0, scale=1, size=(N,))
    a3 = np.random.normal(loc=0, scale=1, size=(N,))
    a4 = np.random.normal(loc=0, scale=1, size=(N,))

    Xa = np.vstack([a1, a2, a3, a4]).T
    assert Xa.shape == (N, P)

    if exact:
        b1 = a1  + np.ones(N)*2
        b2 = a2  + np.ones(N)*3
        b3 = -a3 + np.ones(N)*4
    else:
        b1 = a1  + np.random.normal(loc=0, scale=0.2, size=(N,))
        b2 = a2  + np.random.normal(loc=0, scale=0.4, size=(N,))
        b3 = -a3 + np.random.normal(loc=0, scale=0.3, size=(N,))

    Xb = np.vstack([b1, b2, b3]).T
    assert Xb.shape == (N, Q)

    return Xa, Xb
