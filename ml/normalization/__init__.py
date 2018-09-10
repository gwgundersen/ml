"""=============================================================================
Normalization utility functions.
============================================================================="""

def feature_scaling(X):
    """
    :return: X' = (X - X_min) / (X_max - X_min)
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)

# ------------------------------------------------------------------------------

def mean_center(X):
    """
    :return: X' = X - mean(X)
    """
    return X - X.mean(axis=0)