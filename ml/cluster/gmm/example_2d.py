"""=============================================================================
Visualize a Gaussian mixture model.
============================================================================="""

import matplotlib.pyplot as plt
import numpy as np
from   scipy.stats import multivariate_normal

xi = np.arange(-2, 2, 0.01)
yi = np.arange(-2, 2, 0.01)
xi, yi = np.meshgrid(xi, yi)

x = np.vstack([xi.flatten(), yi.flatten()]).T
z = multivariate_normal.pdf(x, mean=np.array([0.25, 0.3]), cov=np.array([[1, 0], [0.9, 1]]))
zi = z.reshape(xi.shape)

plt.gca().set_aspect('equal', adjustable='box')
plt.contour(xi, yi, zi)
plt.show()
