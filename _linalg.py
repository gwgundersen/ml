"""================================================åå============================
Linear algebra utility functions.
============================================================================"""

import numpy as np
from   scipy import linalg

# -----------------------------------------------------------------------------


def mm(A, B):
    return np.matmul(A, B)


def sqrt(X):
    return linalg.sqrtm(X)


def inv(X):
    return np.linalg.inv(X)


def cholesky(X):
    return np.linalg.cholesky(X)


def svd(X, **kwargs):
    return np.linalg.svd(X, **kwargs)


def eig(X):
    return np.linalg.eig(X)


def rank(X):
    return np.linalg.matrix_rank(X)


def cov(X, Y=None):
    """
    Compute covariance matrix for X or cross covarianc ematrix for X and Y.

    :param X: 2-dimensional matrix.
    :param Y: (Optional) 2-dimensional matrix.
    :return:  Covariance matrix for X or cross-covariance matrix for X and Y.
    """
    assert len(X.shape) == 2
    if type(Y) is np.ndarray:
        return _cross_cov(X, Y)

    N, _ = X.shape
    X_ = X - X.mean(axis=0)
    normalizer = N - 1
    result = np.dot(X_.T, X_) / normalizer
    assert np.isclose(result, np.cov(X.T)).all()
    return result


def _cross_cov(X, Y):
    """
    Compute cross-covariance matrix for X and Y.

    :param X: 2-dimensional matrix.
    :param Y: (Optional) 2-dimensional matrix.
    :return:  Cross-covariance matrix for X and Y.
    """
    assert X.shape[0] == Y.shape[0]
    N, _ = X.shape

    X_ = X - X.mean(axis=0)
    Y_ = Y - Y.mean(axis=0)
    normalizer = N - 1

    Cxy = np.block([
        [cov(X), np.matmul(X_.T, Y_) / normalizer],
        [np.matmul(Y_.T, X_) / normalizer, cov(Y)]
    ])

    assert np.isclose(Cxy, np.cov(X.T, Y.T)).all()
    return Cxy


def corr(X):
    """
    Compute correlation matrix for X.

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


def norm_columns(X):
    """
    Normalize columns of X.

    :param X: Matrix to normalize.
    :return:  Normalized matrix.
    """
    X /= np.linalg.norm(X, 2, axis=0)
    assert np.allclose(np.linalg.norm(X, 2, axis=0), 1)
    return X
