"""=============================================================================
Probabilistic canonical correlation analysis, a complete example.
============================================================================="""

import matplotlib.pyplot as plt

from   ml.datasets import load_paired
from   ml.decomposition.pcca.model import PCCA

# ------------------------------------------------------------------------------

X1, X2 = load_paired(N=10000, P=2000, Q=2000, exact=False, mean_centered=True)
print('Data loaded')

pcca = PCCA(n_components=2)
pcca.fit([X1, X2], n_iters=10)
X1_, X2_ = pcca.sample(1000)

# fig, ax = plt.subplots()
# ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='.')
# ax.scatter(X2[:, 0], X2[:, 1], c='blue', marker='.')
# ax.scatter(X1_[:, 0], X1_[:, 1], c='orange', marker='*')
# ax.scatter(X2_[:, 0], X2_[:, 1], c='cyan', marker='*')

# plt.show()
