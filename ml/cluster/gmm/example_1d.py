"""=============================================================================
Visualize a Gaussian mixture model.
============================================================================="""

import matplotlib.pyplot as plt
import numpy as np
from   scipy.stats import multivariate_normal

plt.style.use('seaborn-deep')

# ------------------------------------------------------------------------------

LINE_WEIGHT = 2
FONT_SIZE = 16

# ------------------------------------------------------------------------------

x = np.arange(-4, 8, 0.1)

# Create densities p(x) and mixture weights \pi_k.
means = [-2, 1, 4]
sigmas = [0.5, 2, 1]
densities = np.array([
    multivariate_normal.pdf(x, mean=-2, cov=0.5),
    multivariate_normal.pdf(x, mean=1,  cov=2),
    multivariate_normal.pdf(x, mean=4,  cov=1)
])
weights = np.array([1/3., 1/3., 1/3.])

# Sanity check configuration.
assert len(densities) == len(weights)
assert np.isclose(weights.sum(), 1.0)

plt.figure(figsize=(8, 5))

legend = []
mixture = []
for i, (d, w, c, mu, sigma) in enumerate(zip(densities, weights,
                                             ['r', 'g', 'b'], means, sigmas)):
    plt.plot(x, w * d, linewidth=LINE_WEIGHT, ls='--', c=c)
    mixture.append(w * d)
    legend.append('mean = %s, var = %s, weight = %s' % (mu, sigma, round(w, 2)))

legend.append('GMM density')
mixture = np.array(mixture).sum(axis=0)
plt.plot(x, mixture, linewidth=LINE_WEIGHT, color='k')

plt.xlabel('x',    fontsize=18)
plt.ylabel('p(x)', fontsize=18)
# plt.legend(legend)
plt.savefig('ml/cluster/gmm/example_1d.png')
# plt.show()
