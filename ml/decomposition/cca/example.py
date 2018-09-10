"""=============================================================================
Canonical correlation analysis, a complete example.
============================================================================="""

import matplotlib.pyplot as plt
import numpy as np

from   ml.datasets import load_paired
from   ml.decomposition.cca.model import CCA

# ------------------------------------------------------------------------------

K = 2
X1, X2 = load_paired(exact=False)
cca = CCA(n_components=K, use_svd=False)
Z1, Z2 = cca.fit_transform(X1, X2)

for i in range(K-1):
    # The embeddings are orthogonal to each other.
    z1a = Z1[:, i]
    z1b = Z1[:, i+1]
    assert np.dot(z1a, z1b)

    z2a = Z2[:, i]
    z2b = Z2[:, i+1]
    assert np.dot(z2a, z2b)

fig, ax = plt.subplots()

z1_0 = Z1[:, 0]
z1_1 = Z1[:, 1]

print(np.linalg.norm(z1_0))
print(np.linalg.norm(z1_1))

plt.plot([0, z1_0[0]], [0, z1_0[1]])
plt.plot([0, z1_1[0]], [0, z1_1[1]])

ax.axis('equal')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()

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