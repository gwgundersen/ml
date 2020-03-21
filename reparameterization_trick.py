"""============================================================================
Simple example of the reparameterization trick.
============================================================================"""

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------

dim = 2
n   = 1000
mu  = np.zeros(dim)
Sig = np.eye(dim)
Sig[1, 0] = 0.9
Sig[0, 1] = 0.9

fig, ax = plt.subplots()
fig.set_size_inches(5, 5)

for _ in range(n):
    s = np.random.multivariate_normal(mean=mu, cov=Sig)
    ax.scatter(s[0], s[1], c='red', marker='.')

for _ in range(n):
    eps = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    L = np.linalg.cholesky(Sig)
    s = mu + np.matmul(L, eps)
    ax.scatter(s[0], s[1], c='blue', marker='.')

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()
