"""=============================================================================
Library interface.
============================================================================="""

import numpy as np
from sklearn.decomposition import FactorAnalysis

mm = np.matmul

# ------------------------------------------------------------------------------

N    = 10000
mean = np.array([0, 0, 0])
cov  = np.array([[1, 0.9, 0],
                 [0.9, 1, 0],
                 [0, 0, 1]])
X = np.random.multivariate_normal(mean, cov, size=N)

print('=' * 80)
print('Sanity check:\n')
print(np.cov(X.T).round(2))
print(X.mean(axis=0).round(2))

# ------------------------------------------------------------------------------

print('=' * 80)
print('Factor analysis:\n')
fa = FactorAnalysis(n_components=2)
fa.fit(X)
mean = np.array([0, 0])
cov  = np.array([[1, 0],
                 [0, 1]])
Z = np.random.multivariate_normal(mean, cov, size=N)

# Generate new samples and look at their statistics.
L  = fa.components_.T    # 3 x 2
u  = fa.noise_variance_  # 3 x 1
X_ = mm(L, Z.T).T + u    # N x 2

print('Loadings:')
print(L.round(2))

import pdb; pdb.set_trace()

print('\nFantasy data:')
print(np.cov(X_.T).round(2))
print(X_.mean(axis=0).round(2))

# ------------------------------------------------------------------------------
