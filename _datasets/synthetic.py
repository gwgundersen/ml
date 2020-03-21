"""=============================================================================
Synthetic dataset.
============================================================================="""

import numpy as np

# ------------------------------------------------------------------------------

N_SAMPLES = 500
densities = [
    {
        'mu':     np.array([0.15, 0.35]),
        'sigma':  np.array([[0.01, 0.0125],
                            [0.0125, 0.01]]),
        'weight': 0.5
    },
    {
        'mu':     np.array([0.5, 0.5]),
        'sigma':  np.array([[0.01, -0.0125],
                            [-0.0125, 0.01]]),
        'weight': 0.3
    },
    {
        'mu':     np.array([0.9, 0.5]),
        'sigma':  np.array([[0.01, 0.0125],
                            [0.0125, 0.01]]),
        'weight': 0.2
    }
]

# ------------------------------------------------------------------------------

def load():
    samples = np.zeros((N_SAMPLES, 2))
    k = 0
    labels = np.zeros(N_SAMPLES)
    for j, params in enumerate(densities):
        mu, sigma, weight = params['mu'], params['sigma'], params['weight']
        n_samples = int(weight * N_SAMPLES)
        for i in range(n_samples):
            samples[k] = np.random.multivariate_normal(mu, sigma)
            labels[k] = j
            k += 1
    return samples, labels
