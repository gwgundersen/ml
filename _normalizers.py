"""============================================================================
Normalization utility functions.
============================================================================"""


def feature_scaling(X):
    """
    :return: X with each column in the range [0, 1].
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)


def mean_center(X):
    """
    :return: X with each column mean centered.
    """
    return X - X.mean(axis=0)


def normalize_columns(X):
    """
    :return: X with each column normalized.

    Example
    -------
    >>> X = np.array([[1000,  10,   0.5],
                     [ 765,   5,  0.35],
                     [ 800,   7,  0.09]])
    >>> normalize_columns(X)
    [[ 1.     1.     1.   ]
     [ 0.765  0.5    0.7  ]
     [ 0.8    0.7    0.18 ]]
    """
    return X / X.max(axis=0)
