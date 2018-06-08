"""=============================================================================
K-means. See Bishop 9.1.
============================================================================="""

import numpy as np
from   ml.cluster.kmeans import viz

# ------------------------------------------------------------------------------

class KMeans():

    def __init__(self, n_components, init_means=None):
        """
        :param n_components: The number of clusters K.
        :param init_means:   Initial centroids if desired.
        :return:             None.
        """
        self.K = n_components
        self.means = init_means

# ------------------------------------------------------------------------------

    def fit(self, X, n_iters=5):
        """
        Iteratively assign data points to their nearest centroids and then
        update the cluster centroids based on the latest assignments.

        This algorithm is computationally intractable because we need to compute
        the Euclidean distance N*K times on each iteration. And we have no
        guarantee of convergence.

        :param X:        N data points with dimensionality D.
        :param n_iters:  Number of iterations to run algorithm.
        :return:         None.
        """
        N, D = X.shape

        # Randomly initialize means.
        if type(self.means) is not np.ndarray:
            self.means = np.random.uniform(X.min(), X.max(), size=(self.K, D))
        self.assignments = np.random.randint(0, self.K, size=(N))

        preds = np.zeros(X.shape[0])
        viz.plot_clusters(X, self.means, preds, _fpath(0), True, init=True)
        Js = []

        for i in range(n_iters):
            self._e_step(X)
            viz.plot_clusters(X, self.means, self.predict(X),
                              _fpath(i + 1, 'a'), True)
            self._m_step(X)
            viz.plot_clusters(X, self.means, self.predict(X),
                              _fpath(i + 1, 'b'), False)
            Js.append(self._cost(X))

        viz.plot_learning_curve(Js, _fpath('learning_curve'))

# ------------------------------------------------------------------------------

    def predict(self, X):
        """
        :param X_: N data points with dimensionality D.
        :return:   An N-vector of integers in [0, K] representing predicted
                   cluster assignments.
        """
        N, _ = X.shape

        predictions = np.zeros(N)
        for n, x in enumerate(X):
            min_dist = np.inf
            for k, centroid in enumerate(self.means):
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

        :param X: N data points with dimensionality D.
        :return:  None.
        """
        for n, x in enumerate(X):
            min_dist = np.inf
            for k, mean in enumerate(self.means):
                # Bishop, eq 9.2.
                dist = self._dist(x, mean)
                if dist <= min_dist:
                    min_dist   = dist
                    new_assign = k
            self.assignments[n] = new_assign

# ------------------------------------------------------------------------------

    def _m_step(self, X):
        """
        For each cluster, re-compute the mean based on the latest assignments.

        :param X: N data points with dimensionality D.
        :return: None
        """
        for k in range(self.K):
            inds = self.assignments == k
            Xk   = X[inds]
            if Xk.size == 0:
                continue
            # Bishop, eq 9.4.
            self.means[k] = Xk.sum(axis=0) / len(Xk)

# ------------------------------------------------------------------------------

    def _cost(self, X):
        """
        :param X: N data points with dimensionality D.
        :return:  Total loss for K-means (square of Euclidean norm).
        """
        J = 0
        for k, mean in zip(range(self.K), self.means):
            inds = self.assignments == k
            Xk   = X[inds]
            for x in Xk:
                J += self._dist(x, mean)
        return J

# ------------------------------------------------------------------------------

    def _dist(self, x1, x2):
        """
        :param x1: A data point.
        :param x2: A data point.
        :return:   Squared Euclidean distance between the two data points.
        """
        l2 = np.linalg.norm(x1 - x2)
        return l2**2

# ------------------------------------------------------------------------------

def _fpath(*kwargs):
    """Utility function for naming figures.
    """
    name = '_'.join([str(i) for i in kwargs])
    return 'ml/cluster/kmeans/figures/%s.png' % name