"""=============================================================================
K-means. See Bishop 9.1.
============================================================================="""

import numpy as np
from   ml import viz

# ------------------------------------------------------------------------------

class KMeans():

    def __init__(self, n_components, figs_dir=None):
        """
        :param n_components: The number of clusters K.
        :param figs_dir:     Directory in which to save intermediate figures of
                             the algorithms' progress.
        :return:             None.
        """
        self.K = n_components
        self.figs_dir = figs_dir

# ------------------------------------------------------------------------------

    def fit(self, X, n_iters=1000):
        """
        Iteratively assign data points to their nearest centroids and then
        update the cluster centroids based on the latest assignments.

        This algorithm is computationally intractable because we need to compute
        the Euclidean distance N*K times on each iteration. And we have no
        guarantee of convergence.

        :param X:        N observations with dimensionality D.
        :param n_iters:  Number of iterations to run algorithm.
        :return:         None.
        """
        N, D = X.shape

        # Randomly initialize means.
        self.centroids   = np.random.uniform(X.min(), X.max(), size=(self.K, D))
        self.assignments = np.random.randint(0, self.K, size=(N))

        for i in range(n_iters):
            self._e_step(X)
            if i % 100 == 0 and self.figs_dir is not None:
                fname = '%s/kmeans_%s.png' % (self.figs_dir, i)
                preds = self.predict(X)
                viz.plot_kmeans(X, self.centroids, preds, fname)
            self._m_step(X)

# ------------------------------------------------------------------------------

    def predict(self, X):
        """
        :param X_: N observations with dimensionality D.
        :return:   An N-vector of integers in [0, K] representing predicted
                   cluster assignments.
        """
        N, _ = X.shape

        predictions = np.zeros(N)
        for n, x in enumerate(X):
            min_dist = np.inf
            for k, centroid in enumerate(self.centroids):
                dist = np.linalg.norm(x - centroid)
                if dist <= min_dist:
                    min_dist = dist
                    pred     = k
            predictions[n] = pred
        return predictions

# ------------------------------------------------------------------------------

    def _e_step(self, X):
        """
        For each data point, assign it to the cluster with the closest mean.

        :param X: N observations with dimensionality D.
        :return:  None.
        """
        for n, x in enumerate(X):
            min_dist = np.inf
            for k, centroid in enumerate(self.centroids):
                # Bishop, eq 9.2.
                dist = np.linalg.norm(x - centroid)
                if dist <= min_dist:
                    min_dist   = dist
                    new_assign = k
            self.assignments[n] = new_assign

# ------------------------------------------------------------------------------

    def _m_step(self, X):
        """
        For each cluster, re-compute the mean based on the latest assignments.

        :param X: N observations with dimensionality D.
        :return: None
        """
        for k in range(self.K):
            inds = self.assignments == k
            Xk   = X[inds]
            if Xk.size == 0:
                continue
            # Bishop, eq 9.4.
            self.centroids[k] = Xk.sum(axis=0) / len(Xk)
