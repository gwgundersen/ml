"""=============================================================================
Linear algebra utility functions.
============================================================================="""

import numpy as np

# ------------------------------------------------------------------------------

def cov(X):
    """
    :param X: 2-dimensional matrix.
    :return:  Covariance matrix for X.
    """
    assert len(X.shape) == 2
    N, _ = X.shape
    X_ = X - X.mean(axis=0)
    normalizer = N - 1
    result = np.dot(X_.T, X_) / normalizer
    assert np.isclose(result, np.cov(X[:, 0], X[:, 1])).all()
    return result

# ------------------------------------------------------------------------------

def corr(X):
    """
    :param X: 2-dimensional matrix.
    :return:  Correlation matrix for X.
    """
    assert len(X.shape) == 2
    _, D = X.shape
    sig = cov(X)
    diag = np.eye(D) * np.diag(sig)
    tmp = np.sqrt(np.linalg.inv(diag))
    result = np.matmul(tmp, np.matmul(sig, tmp))
    assert np.isclose(result, np.corrcoef(X.T)).all()
    return result

