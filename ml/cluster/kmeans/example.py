"""=============================================================================
K-means, a complete example.
============================================================================="""

import numpy as np
from   ml import datasets
from   ml.cluster import KMeans

# ------------------------------------------------------------------------------

X = datasets.load_oldfaithful()
# Initialize means  illustrative purposes.
centroids = np.array([[0.2, 0.8], [0.8, 0.2]])
kmeans = KMeans(n_components=2, init_means=centroids)
kmeans.fit(X)