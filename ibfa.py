"""=============================================================================
Inter-battery factor analysis. See:

    Generative models that discover dependencies between data sets
    https://research.cs.aalto.fi/pml/papers/mlsp06.pdf

    For EM updates, see:
    http://gregorygundersen.com/blog/2020/10/25/em-gaussian-factor-models/
============================================================================="""

import numpy as np
from   numpy.linalg import inv


# ------------------------------------------------------------------------------

class IBFA:

    def __init__(self, n_components, n_iters):
        """Initialize IBFA model.
        """
        self.k = n_components
        self.n_iters = n_iters

    def fit(self, X1, X2):
        """Fit model via expectation-maximization.
        """
        self._init_params(X1, X2)
        for _ in range(self.n_iters):
            self._em_step()

    def transform(self, X1, X2):
        """Embed data using fitted model.
        """
        X = np.hstack([X1, X2]).T
        B1, B2, Psi = self._untile_params()

        # Calculate posterior mean of shared latent variables.
        Psi_inv = inv(Psi)
        M = inv(np.eye(self.k) + self.W.T @ Psi_inv @ self.W)
        Z = M @ self.W.T @ Psi_inv @ X

        # Caluate posterior mean of Z1.
        W1 = self.W[:self.p1]
        Psi_inv = inv(W1 @ W1.T + self.var1 * np.eye(self.p1))
        M  = inv(np.eye(self.k) + B1.T @ Psi_inv @ B1)
        Z1 = M @ B1.T @ Psi_inv @ X1.T

        # Calculate posterior mean of Z2.
        W2 = self.W[self.p1:]
        Psi_inv = inv(W2 @ W2.T + self.var2 * np.eye(self.p2))
        M  = inv(np.eye(self.k) + B2.T @ Psi_inv @ B2)
        Z2 = M @ B2.T @ Psi_inv @ X2.T

        return Z.T, Z1.T, Z2.T

    def fit_transform(self, X1, X2):
        self.fit(X1, X2)
        return self.transform(X1, X2)

    def sample(self):
        """Sample from the fitted model.
        """
        # Calculate posterior mean of Z.
        B1, B2, Psi = self._untile_params()
        Psi_inv   = inv(Psi)
        M = inv(np.eye(self.k) + self.W.T @ Psi_inv @ self.W)
        Z = M @ self.W.T @ Psi_inv @ self.X

        # Caluate posterior mean of Z1.
        W1 = self.W[:self.p1]
        Psi_inv = inv(W1 @ W1.T + self.var1 * np.eye(self.p1))
        M  = inv(np.eye(self.k) + B1.T @ Psi_inv @ B1)
        Z1 = M @ B1.T @ Psi_inv @ self.X1.T

        # Calculate posterior mean of Z2.
        W2 = self.W[self.p1:]
        Psi_inv = inv(W2 @ W2.T + self.var2 * np.eye(self.p2))
        M  = inv(np.eye(self.k) + B2.T @ Psi_inv @ B2)
        Z2 = M @ B2.T @ Psi_inv @ self.X2.T

        # Gaussian mean
        X1_mean = W1 @ Z + B1 @ Z1
        X2_mean = W2 @ Z + B2 @ Z2
        n_samples = X1_mean.shape[1]
        assert(X2_mean.shape[1] == n_samples)

        Psi1 = self.var1 * np.eye(self.p1)
        Psi2 = self.var2 * np.eye(self.p2)

        X1 = np.empty((n_samples, self.p1))
        X2 = np.empty((n_samples, self.p2))

        for i in range(n_samples):
            X1[i] = np.random.multivariate_normal(X1_mean[:, i], Psi1)
            X2[i] = np.random.multivariate_normal(X2_mean[:, i], Psi2)

        return X1, X2

# ------------------------------------------------------------------------------

    def _em_step(self):
        """Perform EM on parameters W, B, and variances.
        """
        B1, B2, Psi = self._untile_params()
        Psi_inv     = inv(Psi)

        # Update W.
        M     = inv(np.eye(self.k) + self.W.T @ Psi_inv @ self.W)
        A     = M @ self.W.T @ Psi_inv
        S     = inv(M + A @ self.Sigma @ A.T)
        W_new = self.Sigma @ A.T @ S

        # Update B1.
        W1      = W_new[:self.p1]
        Psi_inv = inv(W1 @ W1.T + self.var1 * np.eye(self.p1))
        M       = inv(np.eye(self.k) + B1.T @ Psi_inv @ B1)
        A       = M @ B1.T @ Psi_inv
        Sigma   = self.Sigma[:self.p1, :self.p1]
        B1_new  = Sigma @ A.T @ inv(M + A @ Sigma @ A.T)

        # Update var1.
        V        = Sigma - Sigma @ A.T @ B1.T - W1 @ W1.T
        var1_new = (1/self.p1) * np.trace(V)

        # Update B2.
        W2      = W_new[self.p1:]
        Psi_inv = inv(W2 @ W2.T + self.var2 * np.eye(self.p2))
        M       = inv(np.eye(self.k) + B2.T @ Psi_inv @ B2)
        A       = M @ B2.T @ Psi_inv
        Sigma   = self.Sigma[self.p1:, self.p1:]
        B2_new  = Sigma @ A.T @ inv(M + A @ Sigma @ A.T)

        # Update var2.
        V        = Sigma - Sigma @ A.T @ B2.T - W2 @ W2.T
        var2_new = (1/self.p2) * np.trace(V)

        # Update state.
        self.W    = W_new
        self.B    = np.vstack([B1_new, B2_new])
        self.var1 = var1_new
        self.var2 = var2_new

    def _untile_params(self):
        """Utility functions for constructing B1, B2, and Psi.
        """
        B1 = self.B[:self.p1]
        B2 = self.B[self.p1:]

        Psi1 = self.var1 * np.eye(self.p1) + B1 @ B1.T
        Psi2 = self.var2 * np.eye(self.p2) + B2 @ B2.T
        Psi  = np.block([[Psi1, np.zeros((self.p1, self.p2))],
                         [np.zeros((self.p2, self.p1)), Psi2]])

        return B1, B2, Psi

    def _init_params(self, X1, X2):
        """Initialize parameters.
        """
        self.X1, self.X2 = X1, X2
        self.n, self.p1  = self.X1.shape
        n2, self.p2      = self.X2.shape
        self.p           = self.p1 + self.p2
        assert(self.n == n2)

        # Initialize sample covariances matrices.
        self.X     = np.hstack([X1, X2]).T
        Sigma1     = np.cov(self.X1.T)
        Sigma2     = np.cov(self.X2.T)
        self.Sigma = np.block([[Sigma1, np.zeros((self.p1, self.p2))],
                               [np.zeros((self.p2, self.p1)), Sigma2]])
        assert(self.X.shape == (self.p,  self.n))
        assert(Sigma1.shape == (self.p1, self.p1))
        assert(Sigma2.shape == (self.p2, self.p2))

        # Initialize W.
        W1     = np.random.random((self.p1, self.k))
        W2     = np.random.random((self.p2, self.k))
        self.W = np.vstack([W1, W2])
        assert(self.W.shape == (self.p, self.k))

        # Initialize B.
        B1     = np.random.random((self.p1, self.k))
        B2     = np.random.random((self.p2, self.k))
        self.B = np.vstack([B1, B2])
        assert(self.B.shape == (self.p, self.k))

        # Initialize variances.
        self.var1 = 1
        self.var2 = 1


# ------------------------------------------------------------------------------
# Example.
# ------------------------------------------------------------------------------

from   io import StringIO
import matplotlib.pyplot as plt
import pandas as pd
import os
import urllib.request


if not os.path.exists('sales.txt'):
    # Download dataset if needed. See
    #
    #   https://online.stat.psu.edu/stat505/lesson/13/13.2
    #
    # for a dataset description.
    url = 'https://online.stat.psu.edu/'\
          'onlinecourses/sites/stat505/files/data/sales.txt'
    with urllib.request.urlopen(url) as f:
        text = f.read().decode('utf-8')
        df = pd.read_csv(StringIO(text), sep='  ')
else:
    df = pd.read_csv('sales.txt', sep='  ')


df = pd.DataFrame(data=df.values, columns=range(7))
X1 = df[[0, 1, 2]].values
X2 = df[[3, 4, 5, 6]].values

# Normalize data.
X1 = X1 - X1.mean(axis=0)
X1 = X1 / X1.std(axis=0)
X2 = X2 - X2.mean(axis=0)
X2 = X2 / X2.std(axis=0)

ibfa      = IBFA(n_components=1, n_iters=1000)
Z, Z1, Z2 = ibfa.fit_transform(X1, X2)
X1i, _    = ibfa.sample()

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(14, 7)

# Sanity check data generating process.
ax1.set_title('Data vs. generated samples')
ax1.scatter(X1[:, 0],  X1[:, 1],  label=r'true $X_1$')
ax1.scatter(X1i[:, 0], X1i[:, 1], label=r'inferred $X_1$')
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
ax1.legend()

# Check for correlation between LV1 and sales growth.
inds = Z[:, 0].argsort()
ax2.set_title('Latent variable 1')
ax2.scatter(X1[inds][:, 0], Z[inds][:, 0])
ax2.set_xlabel('LV1')
ax2.set_ylabel('Sales growth')

plt.show()
