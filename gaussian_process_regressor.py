"""============================================================================
Gaussian process regressor.
============================================================================"""

import numpy as np
from   scipy.spatial.distance import pdist, cdist, squareform
from   scipy.linalg import cholesky, cho_solve


# ------------------------------------------------------------------------------

class GPRegressor:

    def __init__(self, length_scale=1):
        self.length_scale = length_scale
        # In principle, this could be configurable.
        self.kernel = rbf_kernel

    def fit(self, X, y):
        self.kernel_ = self.kernel(X, length_scale=self.length_scale)
        lower = True
        L = cholesky(self.kernel_, lower=lower)
        self.alpha_ = cho_solve((L, lower), y)
        self.X_train_ = X
        self.L_ = L

    def predict(self, X):
        K_star = self.kernel(X, self.X_train_, length_scale=self.length_scale)
        y_mean = K_star.dot(self.alpha_)
        lower = True
        v = cho_solve((self.L_, lower), K_star.T)
        y_cov = self.kernel(X, length_scale=self.length_scale) - K_star.dot(v)
        return y_mean, y_cov


# ------------------------------------------------------------------------------

def rbf_kernel(X, Y=None, length_scale=1):
    if Y is None:
        dists = pdist(X / length_scale, metric='sqeuclidean')
        K = np.exp(-.5 * dists)
        K = squareform(K)
        np.fill_diagonal(K, 1)
    else:
        dists = cdist(X / length_scale, Y / length_scale, metric='sqeuclidean')
        K = np.exp(-.5 * dists)
    return K
