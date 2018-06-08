"""=============================================================================
Gaussian mixture model, a complete example.
============================================================================="""

import matplotlib.pyplot as plt
import numpy as np

from   ml.cluster.gmm.model import GMM
from   ml import datasets
from   ml.cluster.gmm import viz

# ------------------------------------------------------------------------------

np.random.seed(0)

X   = datasets.load_oldfaithful()
gmm = GMM(n_components=2)
gmm.fit(X)

plt.clf()
n_samples = 100
X_sim, Y_sim = gmm.sample(X.shape[1], n_samples)
x = X_sim[:, 0]
y = X_sim[:, 1]
plt.scatter(x, y, c=Y_sim)

# plt.xlim(x.min(), x.max())
# plt.ylim(y.min(), y.max())

plt.savefig('ml/cluster/gmm/figures/fantasy.png')
