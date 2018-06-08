"""=============================================================================
Probabilistic PCA, a complete example.
============================================================================="""

from ml.datasets import load_iris
from ml.decomposition.ppca.model import PPCA
from ml.decomposition.ppca import viz

# ------------------------------------------------------------------------------

X, Y = load_iris()
ppca = PPCA(var=0.1)
ppca.fit(X)
viz.plot_scatter(ppca.transform(), Y)