"""=============================================================================
Probabilistic canonical correlation analysis, a complete example.
============================================================================="""

import matplotlib.pyplot as plt

from   ml.datasets import load_paired
from   ml.decomposition.pcca.model import PCCA
from ml.decomposition.cca.model import CCA

# ------------------------------------------------------------------------------

X1, X2 = load_paired(exact=False)

pcca = PCCA(n_components=2)
pcca.fit_transform(X1, X2)

# Z1, Z2 = pcca.fit_transform(X1, X2)
#
# fig, ax = plt.subplots()
# ax.scatter(Z1[:, 0], Z1[:, 1], c='blue')
# ax.scatter(Z2[:, 0], Z2[:, 1], c='red')
#
# for i, txt in enumerate(range(len(Z1))):
#     ax.annotate(txt, (Z1[:, 0][i], Z1[:, 1][i]), color='blue')
#
# for i, txt in enumerate(range(len(Z2))):
#     ax.annotate(txt, (Z2[:, 0][i], Z2[:, 1][i]), color='red')
#
# plt.show()