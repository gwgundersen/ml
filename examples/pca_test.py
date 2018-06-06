import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds

from ml.decomposition.pca.ppca import PPCA


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

# ------------------------------------------------------------------------------

iris         = ds.load_iris()
iris_y       = np.transpose(iris.data)
iris_classes = iris.target

ppca = PPCA(var=0.1)
# ppca.fit(iris_y)
ppca.fit(iris_y)

plot_scatter(ppca.transform(), iris_classes)
plt.show()