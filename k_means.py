"""============================================================================
K-means. See Bishop 9.1.
============================================================================"""


class KMeans:

    def __init__(self, n_components, init_means=None):
        """
        :param n_components: The number of clusters K.
        :param init_means:   Initial centroids if desired.
        :return:             None.
        """
        self.K = n_components
        self.means = init_means

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
        Js = []

        for i in range(n_iters):
            self._e_step(X)
            self._m_step(X)
            Js.append(self._cost(X))

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

    def _dist(self, x1, x2):
        """
        :param x1: A data point.
        :param x2: A data point.
        :return:   Squared Euclidean distance between the two data points.
        """
        l2 = np.linalg.norm(x1 - x2)
        return l2**2


# -----------------------------------------------------------------------------
# Example.
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    from   _datasets import load_oldfaithful
    import matplotlib.pyplot as plt

    def _midpoint(p1, p2):
        return (p1 + p2) / 2.0

    def plot_clusters(X, centroids, preds, fname):
        """
        Plot 2D K-means clusters.
        """
        assert centroids.shape == (2, 2)

        x = X[:, 0]
        y = X[:, 1]

        colors = ['r' if p == 0 else 'b' for p in preds]
        plt.scatter(x, y, c=colors)

        c1, c2 = centroids

        cx, cy = c1
        plt.scatter(cx, cy, marker='x', facecolor='w', linewidth=6, s=150)
        plt.scatter(cx, cy, marker='x', facecolor='#800000', linewidth=2,
                    s=100)
        cx, cy = c2
        plt.scatter(cx, cy, marker='x', facecolor='w', linewidth=6, s=150)
        plt.scatter(cx, cy, marker='x', facecolor='#000080', linewidth=2,
                    s=100)

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

    X = load_oldfaithful()
    # Initialize means  illustrative purposes.
    centroids = np.array([[0.2, 0.8], [0.8, 0.2]])
    kmeans = KMeans(n_components=2, init_means=centroids)
    kmeans.fit(X)

    plot_clusters(X, kmeans.means, kmeans.predict(X), '_figures/kmeans.png')
