"""=============================================================================
Utility functions for visualizing GMMs.
============================================================================="""

from   matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-deep')

# ------------------------------------------------------------------------------

def plot_log_likelihood(Y, fname):
    """
    Plot GMM's log-likelihood curve.
    """
    X = range(len(Y))
    plt.gca().set_aspect('auto')
    plt.plot(X, Y)
    plt.savefig(fname)

# ------------------------------------------------------------------------------

def plot_gmm(X, Y_, means, covariances, fname):

    fig, ax = plt.subplots(1, 1)
    colors = ['r', 'b']

    for i, (mean, covar) in enumerate(zip(means, covariances)):
        v, w = np.linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])

        # As the DP will not use every component it has access to unless it
        # needs it, we shouldn't plot the redundant components.
        if not np.any(Y_ == i):
            continue
        if u[0] == 0:
            continue

        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], color=colors[i])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = Ellipse(mean, v[0], v[1], 180. + angle, color=colors[i])
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)

        ax.add_artist(ell)

    x = X[:, 0]
    y = X[:, 1]
    eps = 0.1
    plt.xlim(x.min() - eps, x.max() + eps)
    plt.ylim(y.min() - eps, y.max() + eps)

    plt.locator_params(nbins=4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(fname)
    plt.clf()

