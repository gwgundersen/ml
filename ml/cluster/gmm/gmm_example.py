"""=============================================================================
Visualize a Gaussian mixture model.
============================================================================="""

import matplotlib.pyplot as plt
import numpy as np
from   scipy.stats import multivariate_normal

# ------------------------------------------------------------------------------

LINE_WEIGHT = 2
FONT_SIZE = 16

# ------------------------------------------------------------------------------

x = np.arange(-4, 8, 0.1)

# Create densities p(x) and mixture weights \pi_k.
densities = np.array([
    multivariate_normal.pdf(x, mean=-2, cov=0.5),
    multivariate_normal.pdf(x, mean=1,  cov=2),
    multivariate_normal.pdf(x, mean=4,  cov=1)
])
weights = np.array([0.5, 0.2, 0.3])

# Sanity check configuration.
assert len(densities) == len(weights)
assert np.isclose(weights.sum(), 1.0)

plt.figure(figsize=(8, 8))

legend = []
mixture = []
for i, (d, w) in enumerate(zip(densities, weights)):
    plt.plot(x, w * d, linewidth=LINE_WEIGHT, ls='--')
    mixture.append(w * d)
    legend.append('Component %s, pi = %s' % (i+1, round(w, 2)))

legend.append('GMM density')
mixture = np.array(mixture).sum(axis=0)
plt.plot(x, mixture, linewidth=LINE_WEIGHT, color='k')

plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Mixture of 1D Gaussians', fontsize=FONT_SIZE)
plt.legend(legend)
plt.show()
