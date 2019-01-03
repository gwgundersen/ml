"""=============================================================================
Probabilistic canonical correlation analysis, a complete example.
============================================================================="""

import matplotlib.pyplot as plt
import time

from   ml.datasets import load_paired
from   ml.datasets import load_lowrankcov
from   ml.decomposition.rpcca.model import RPCCA

# ------------------------------------------------------------------------------

k = 10

X1, X2 = load_lowrankcov(N=200, P=50, Q=50, k=k)
print('Data loaded')

# pcca = RPCCA(n_components=2)
# s = time.time()
# pcca.fit([X1, X2], n_iters=1000)
# print('Time to fit: %s' % str(time.time() - s))

# print(pcca.nlls)

pcca = RPCCA(n_components=2, rank_k=k)
s = time.time()
pcca.fit([X1, X2], n_iters=100)
print('Time to fit: %s' % str(time.time() - s))

print('Model fitted')
X1_, X2_ = pcca.sample(200)
print('Sampling done')

fig, ax = plt.subplots()
ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='.')
ax.scatter(X2[:, 0], X2[:, 1], c='blue', marker='.')
ax.scatter(X1_[:, 0], X1_[:, 1], c='orange', marker='*')
ax.scatter(X2_[:, 0], X2_[:, 1], c='cyan', marker='*')

# plt.scatter(list(range(len(pcca.nlls))), pcca.nlls)

plt.show()
