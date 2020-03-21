"""============================================================================
Old Faithful eruptions dataset.
============================================================================"""

import numpy as np


# -----------------------------------------------------------------------------

def scale_features(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)


def load():
    """
    :return: Old faithful reuprtions data with shape (272, 2).
    """
    X = np.loadtxt('_datasets/oldfaithful/data', skiprows=1)[:, 1:]
    return scale_features(X)
