"""============================================================================
Probabilistic canonical correlation analysis. For references in comments:

    A Probabilistic Interpretation of Canonical Correlation Analysis.
    Bach, Jordan (2006).

    The EM algorithm for mixtures of factor analyzers.
    Ghahramani, Hinton (1996).
============================================================================"""

import numpy as np

inv = np.linalg.inv


# -----------------------------------------------------------------------------

class ProbabilisticCCA:

    def __init__(self, n_components, n_iters):
        """Initialize probabilistic CCA model.
        """
        self.k = n_components
        self.n_iters = n_iters

    def fit(self, X1, X2):
        """Fit model via EM.
        """
        self._init_params(X1, X2)
        np.linalg.cholesky(self.Psi)
        print('is psd')
        for _ in range(self.n_iters):
            print(_)
            self._em_step()
            np.linalg.cholesky(self.Psi)

    def transform(self, X1, X2):
        """Embed data using fitted model.
        """
        X = np.hstack([X1, X2]).T
        Psi_inv = inv(self.Psi)
        M = inv(np.eye(self.k) + self.W.T @ Psi_inv @ self.W)
        Z = M @ self.W.T @ Psi_inv @ X
        return Z.T

    def fit_transform(self, X1, X2):
        self.fit(X1, X2)
        return self.transform(X1, X2)

    def sample(self, n_samples):
        """Sample from the fitted model.
        """
        Psi_inv = inv(self.Psi)
        M = inv(np.eye(self.k) + self.W.T @ Psi_inv @ self.W)
        Z_post_mean = M @ self.W.T @ Psi_inv @ self.X

        X_mean = self.W @ Z_post_mean
        assert(X_mean.shape == (self.p, n_samples))
        X = np.zeros((n_samples, self.p))

        for i in range(n_samples):
            X[i] = np.random.multivariate_normal(X_mean[:, i], self.Psi)

        return X[:, :self.p1], X[:, self.p1:]

# -----------------------------------------------------------------------------

    def _em_step(self):
        """Perform EM on parameters W and Psi
        """
        Psi_inv = inv(self.Psi)
        M = inv(np.eye(self.k) + self.W.T @ Psi_inv @ self.W)
        S = M @ self.W.T @ Psi_inv @ self.X
        A = self.n * M + S @ S.T

        W_new = self.X @ S.T @ inv(A)

        W1       = self.W[:self.p1]
        W1_new   = W_new[:self.p1]
        Psi1_inv = Psi_inv[:self.p1, :self.p1]
        Psi1_new = self.Sigma1 - self.Sigma1 @ Psi1_inv @ W1 @ M @ W1_new.T

        W2       = self.W[self.p1:]
        W2_new   = W_new[self.p1:]
        Psi2_inv = Psi_inv[self.p1:, self.p1:]
        Psi2_new = self.Sigma2 - self.Sigma2 @ Psi2_inv @ W2 @ M @ W2_new.T

        Psi_new = np.block([[Psi1_new, np.zeros((self.p1, self.p2))],
                            [np.zeros((self.p2, self.p1)), Psi2_new]])

        self.W   = W_new
        self.Psi = Psi_new

    def _init_params(self, X1, X2):
        """Initialize parameters.
        """
        self.X1, self.X2 = X1, X2
        self.n, self.p1 = self.X1.shape
        _, self.p2 = self.X2.shape
        self.p = self.p1 + self.p2

        # Initialize sample covariances matrices.
        self.X = np.hstack([X1, X2]).T
        assert(self.X.shape == (self.p, self.n))
        self.Sigma1 = np.cov(self.X1.T)
        assert(self.Sigma1.shape == (self.p1, self.p1))
        self.Sigma2 = np.cov(self.X2.T)
        assert(self.Sigma2.shape == (self.p2, self.p2))

        # Initialize W.
        W1 = np.random.random((self.p1, self.k))
        W2 = np.random.random((self.p2, self.k))
        self.W = np.vstack([W1, W2])
        assert(self.W.shape == (self.p, self.k))

        # Initialize Psi.
        prior_var1 = 1
        prior_var2 = 1
        Psi1 = prior_var1 * np.eye(self.p1)
        Psi2 = prior_var2 * np.eye(self.p2)
        Psi = np.block([[Psi1, np.zeros((self.p1, self.p2))],
                        [np.zeros((self.p2, self.p1)), Psi2]])
        self.Psi = Psi


# -----------------------------------------------------------------------------
# Example.
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
from   _datasets import load_lowrankcov


k = 10
X1, X2 = load_lowrankcov(N=200, P=50, Q=40, k=k)

pcca = ProbabilisticCCA(n_components=2, n_iters=100)
pcca.fit(X1, X2)
X1_, X2_ = pcca.sample(200)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(X1[:, 0], X1[:, 1])
ax1.scatter(X2[:, 0], X2[:, 1])
ax2.scatter(X1_[:, 0], X1_[:, 1])
ax2.scatter(X2_[:, 0], X2_[:, 1])

plt.show()
# plt.savefig('_figures/pcca.png')
