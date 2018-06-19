"""=============================================================================
Gaussian mixture model, a complete example.
============================================================================="""

import matplotlib.pyplot as plt
import numpy as np
from   scipy.interpolate import griddata

from   ml.cluster.gmm.model import GMM
from   ml import datasets
from   ml.cluster.gmm import viz

# ------------------------------------------------------------------------------

# Seed 3, n-samples = 1000
np.random.seed(9)

# ------------------------------------------------------------------------------

X, Y = datasets.load_synthetic()

plt.scatter(X[:, 0], X[:, 1], c='#cc00cc')
plt.xlim(-0.25, 1.25)
plt.ylim(-0.25, 1.25)

plt.locator_params(nbins=4)
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
plt.savefig('ml/cluster/gmm/figures/dataset.png')
plt.clf()

# plt.scatter(X[:, 0], X[:, 1], c=Y)

gmm = GMM(n_components=3)
gmm.fit(X)
plt.clf()

print(gmm.weights)

X_sim, Y_sim = gmm.sample(X.shape[1], 100)
colors = ['r' if y == 0 else ('b' if y == 1 else 'g') for y in Y_sim]

plt.scatter(X_sim[:, 0], X_sim[:, 1], c=colors)
plt.xlim(-0.25, 1.25)
plt.ylim(-0.25, 1.25)
plt.locator_params(nbins=4)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('ml/cluster/gmm/figures/fantasy_samples.png')

# ------------------------------------------------------------------------------

plt.clf()

DX = 0.01
xi = X[:, 0]
yi = X[:, 1]
X = np.arange(xi.min(), xi.max(), DX)
Y = np.arange(yi.min(), yi.max(), DX)
X, Y = np.meshgrid(X, Y)
data = np.vstack([X.flatten(), Y.flatten()]).T

Z = gmm.q(data)
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z)

plt.xlim(-0.25, 1.25)
plt.ylim(-0.25, 1.25)
plt.locator_params(nbins=4)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('ml/cluster/gmm/figures/gmm_contours.png')
