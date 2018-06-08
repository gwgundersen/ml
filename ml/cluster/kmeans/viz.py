"""=============================================================================
Utility functions for visualizing K-means.
============================================================================="""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-deep')

# ------------------------------------------------------------------------------

def plot_learning_curve(Y, fname):
    """
    Plot K-means learning curve.
    """
    X = range(len(Y))
    plt.gca().set_aspect('auto')
    plt.plot(X, Y)
    plt.savefig(fname)

# ------------------------------------------------------------------------------

def plot_clusters(X, centroids, preds, fname, plot_hyerplane, init=False):
    """
    Plot 2D K-means clusters.
    """
    assert centroids.shape == (2, 2)

    x = X[:, 0]
    y = X[:, 1]

    if init:
        plt.scatter(x, y, c='g')
    else:
        colors = ['r' if p == 0 else 'b' for p in preds]
        plt.scatter(x, y, c=colors)

    c1, c2 = centroids

    cx, cy = c1
    plt.scatter(cx, cy, marker='x', facecolor='w', linewidth=6, s=150)
    plt.scatter(cx, cy, marker='x', facecolor='#800000', linewidth=2, s=100)
    cx, cy = c2
    plt.scatter(cx, cy, marker='x', facecolor='w', linewidth=6, s=150)
    plt.scatter(cx, cy, marker='x', facecolor='#000080', linewidth=2, s=100)

    if plot_hyerplane:
        # Compute perpendicular hyperplane between two centroids.
        perp = np.zeros((2, 2))
        perp[:, 0] = -1 * centroids[:, 1]
        perp[:, 1] = centroids[:, 0]

         # Shift hyperplane to be positioned at the midpoint.
        c1, c2 = centroids
        pmid = _midpoint(c1, c2)
        p1, p2 = perp
        delta = pmid - p1
        p1 += delta
        p2 += delta

        # Scale the hyperplane to appear infinite.
        scale = p1 - p2
        p1 -= scale * 100
        p2 += scale * 100

        perp_offset = np.vstack([p1, p2])
        plt.plot(perp_offset[:, 0], perp_offset[:, 1], color='#cc00cc',
                 linestyle='dashed')

    eps = 0.1
    plt.xlim(x.min() - eps, x.max() + eps)
    plt.ylim(y.min() - eps, y.max() + eps)

    plt.locator_params(nbins=4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.savefig(fname)
    plt.cla()

# ------------------------------------------------------------------------------

def _midpoint(p1, p2):
    return (p1 + p2) / 2.0