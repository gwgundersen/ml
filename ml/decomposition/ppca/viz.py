"""=============================================================================
Utility functions for visualizing probabilistic PCA.
============================================================================="""

import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------

def plot_scatter(x, classes, ax=None):
    ax = plt.gca() if ax is None else ax
    cmap = plt_cm.jet
    norm = plt_col.Normalize(vmin=np.min(classes), vmax=np.max(classes))
    mapper = plt_cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = mapper.to_rgba(classes)

    x0 = x[0, :]
    x1 = x[1, :]
    ax.set_xlim(xmin=x0.min(), xmax=x0.max())
    ax.set_ylim(ymin=x1.min(), ymax=x1.max())

    ax.scatter(x0, x1, color=colors, s=20)

    plt.savefig('ml/decomposition/ppca/ppca.png')
