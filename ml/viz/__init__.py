"""=============================================================================
Utility functions for visualizing algorithms.
============================================================================="""

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

def plot_kmeans(X, centroids, preds, fname):
    """
    Plot K-means clusters.
    """
    x = X[:, 0]
    y = X[:, 1]

    plt.scatter(x, y, c=preds)
    for c in centroids:
        plt.scatter(c[0], c[1], marker='x', c='k')

    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.savefig(fname)

    plt.cla()
