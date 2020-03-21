"""============================================================================
Probabilistic PCA.

For referenced equations, see Tipping and Bishop (1999).
============================================================================"""

from   numpy import transpose
from   numpy.linalg import inv


# -----------------------------------------------------------------------------

class PPCA:

    def __init__(self, n_components=2, var=1):
        """
        :param n_components: Number of principal components (`q` in T&B).
        :param var:          Prior on the variance.
        :return:             None.
        """
        self.n_components = n_components
        self.prior_var    = var

    def fit(self, X):
        """Fits the probabilistic PCA model parameters using the
        maximum-likelihood estimators.

        :param X: Observations with shape (n_feats, n_samps).
        :return:  None.
        """
        self.X = X
        self.n_feats = X.shape[0]
        self.n_samps = X.shape[1]

        # T&B, paragraph after eq. 5:
        #
        #     The ML estimator for mu is given by the mean of the data.
        #
        mu_ml = np.mean(self.X, 1)[:, np.newaxis]

        # T&B, eq. 5. `ddof=0` returns the average.
        Sigma_hat = np.cov(self.X, ddof=0)

        # T&B, paragraph after eq. 7.
        # TODO: Call `decomposition.svd`.
        U, L, _ = np.linalg.svd(Sigma_hat)

        # This takes the top `n_components` singular values or pads with zeros
        # as needed.
        if self.n_components > len(L):
            tmp = np.zeros(self.n_components)
            tmp[:len(L)] = L
            L = tmp
        else:
            L = L[:self.n_components]

        # T&B, eq. 7.
        tmp = np.sqrt(np.maximum(0, L - self.prior_var))
        W_ml = U[:, :self.n_components].dot(np.diag(tmp))

        # T&B, eq. 8.:
        #
        #     ...which has the clear interpetation as the variance 'lost' in
        #     the projection, averaged over the lost dimensions.
        #
        if self.n_components < self.n_feats:
            norm   = 1.0 / (self.n_feats - self.n_components)
            Sigma_ml = norm * np.sum(L[self.n_components:])
        else:
            Sigma_ml = 0.0

        self.W     = W_ml
        self.mu    = mu_ml
        self.Sigma = Sigma_ml

        return self

    def transform(self, X=None):
        """
        Performs dimensionality reduction with the learned parameters W_ml,
        Sigma_ml, and mu_ml. From T&B, above eq. 6:

            It is more natural from a probabilistic perspective to consider the
            dimensionality-reduction process in terms of the distribution of the
            latent variables, conditioned on the observation.

        See T&B, eq. 6.

        :param X: Observations with shape (n_feats, n_samps).
        :return:  Latent variables with shape (n_components, n_samps).
        """
        if X is None:
            X = self.X
        # T&B, eq. 6.
        M = transpose(self.W).dot(self.W) + self.Sigma * np.eye(self.W.shape[1])
        Z = inv(M).dot(transpose(self.W)).dot(X - self.mu)
        return Z

# ------------------------------------------------------------------------------

    def fit_transform(self, X):
        """
        Fit model and then transforms data using learned model.

        :param X: Observations with shape (n_feats, n_samps).
        :return:  Result of self.transform() function.
        """
        self.fit(X)
        return self.transform()


# ------------------------------------------------------------------------------
# Example.
# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from   _datasets import load_iris
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_col
import numpy as np


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

    plt.savefig('_figures/ppca.png')


X, Y = load_iris()
ppca = PPCA(var=0.1)
ppca.fit(X)
plot_scatter(ppca.transform(), Y)