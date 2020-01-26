"""============================================================================
Bayesian Gaussian mixture model with Gibbs sampling.

Equations come from Blei 2015, "Bayesian Mixture Models and the Gibbs Sampler".

See: http://www.cs.columbia.edu/~blei/fogm/2015F/notes/mixtures-and-gibbs.pdf.
============================================================================"""

import numpy as np
from scipy.stats import multivariate_normal as mvn


# -----------------------------------------------------------------------------

class BayesianGMM:

    def __init__(self, n_components, sigma0=1, lambda0=1):
        self.K = n_components
        self.sigma0 = sigma0
        self.lambda0 = lambda0

    def sample(self, X, n_iters=100):
        """
        Gibbs sampler for mixture of Gaussians.

        :param X: Data with shape (N samples, D predictors).
        :return:  None.
        """
        self.N, self.D = X.shape
        mus, pis = self._init_params()

        for t in range(n_iters):
            Z = self._sample_Z(X, mus, pis)
            mus = self._sample_mu(X, Z)

        self.Z_ = Z
        self.mus_ = mus

    def _sample_Z(self, X, mus, pis):
        """
        """
        Sigma = self.sigma0**2 * np.eye(self.D)
        Z_new = np.empty(self.N)

        for n in range(self.N):
            x_n = X[n]
            prob_Z_n = np.empty(self.K)

            for k in range(self.K):
                log_p = np.log(pis[k]) + mvn.logpdf(x_n, mus[k], Sigma)
                prob_Z_n[k] = np.exp(log_p)

            prob_Z_n /= prob_Z_n.sum()
            Z_new[n] = np.random.choice(self.K, p=prob_Z_n)

        return Z_new

    def _sample_mu(self, X, Z):
        """
        """
        var = self.sigma0**2
        Sigma = var * np.eye(self.D)
        mu_new = np.empty((self.K, self.D))
        for k in range(self.K):
            inds = Z == k
            n_k = inds.sum()
            X_bar_k = X[inds].mean(axis=0)
            mu_hat_k = (n_k / var) / ((n_k / var) + 1) * X_bar_k
            lambda_hat = 1. / (n_k / Sigma + 1 / self.lambda0**2)
            lambda_hat = np.eye(self.D) * lambda_hat
            mu_new[k] = mvn(mu_hat_k, lambda_hat).rvs(size=1)
        return mu_new

    def _init_params(self):
        """
        """
        mu0 = np.zeros(self.D)
        Sigma0 = np.eye(self.D) * self.lambda0
        mus = mvn(mu0, Sigma0).rvs(size=self.K)
        pis = np.ones(self.N) * 1/self.K
        return mus, pis


if __name__ == '__main__':

    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    X, y_true = make_blobs(n_samples=1000, centers=3)
    model = BayesianGMM(n_components=3)
    model.sample(X, n_iters=100)

    plt.scatter(X[:, 0], X[:, 1], c=model.Z_)
    plt.show()
