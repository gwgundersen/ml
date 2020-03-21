"""============================================================================
Probabilistic canonical correlation analysis. For references in comments:

    A Probabilistic Interpretation of Canonical Correlation Analysis.
    Bach, Jordan (2006).

    The EM algorithm for mixtures of factor analyzers.
    Ghahramani, Hinton (1996).
============================================================================"""

import numpy as np

mm     = np.matmul
diag   = np.diag
det    = np.linalg.det
dot    = np.dot
log    = np.log
tr     = np.trace
inv    = np.linalg.inv
Normal = np.random.multivariate_normal


# -----------------------------------------------------------------------------

class PCCA:

    def __init__(self, n_components):
        """
        Initialize the probabilistic CCA model.

        :param n_components: The dimensionality of the latent variable.
        :param rank_k:       The rank to use for low-rank approximation of matrix
                             inversion.
        :return:             None.
        """
        self.n_components = n_components

    def fit(self, X, n_iters):
        """
        Fit the probabilistic CCA model using Expectation-Maximization.

        :param X:       A joint density, represented as a sequence of datasets,
                        each with N observations and p1- and p2-dimensions
                        respectively.
        :param n_iters: The number of EM iterations.
        :return:        None.
        """
        assert type(X) is list or type(X) is tuple
        X1, X2 = X
        n1, p1 = X1.shape
        n2, p2 = X2.shape
        assert n1 == n2
        n = n1
        p = p1 + p2
        k = self.n_components

        X = np.hstack(X).T
        assert X.shape == (p, n)

        nlls = []

        Lambda, Psi = self._init_params(p1, p2)
        for i in range(n_iters):
            # print('Iter: %s' % i)
            Lambda_new, Psi_new = self._em_step(X, Lambda, Psi, n, k)
            nll = self.neg_log_likelihood(X, Lambda, Psi)
            nlls.append(nll)
            Lambda = Lambda_new
            Psi    = Psi_new

        self.X       = X
        self.Lambda  = Lambda
        self.Lambda1 = Lambda[:p1, :]
        self.Lambda2 = Lambda[p1:, :]
        self.Psi     = Psi
        self.Psi1    = Psi[:p1, :p1]
        self.Psi2    = Psi[p1:, p1:]
        self.nlls    = nlls

    def sample(self, n):
        """
        Sample from the fitted probabilistic CCA model.

        :param n: The number of samples.
        :return:  Two views of n samples each.
        """
        Z  = self.E_z_given_x(self.Lambda, self.Psi, self.X)

        m1 = dot(self.Lambda1, Z)
        m2 = dot(self.Lambda2, Z)

        p1, _ = self.Lambda1.shape
        p2, _ = self.Lambda2.shape

        X1 = np.zeros((p1, n))
        X2 = np.zeros((p2, n))

        for i in range(n):
            X1[:, i] = Normal(mean=m1[:, i], cov=self.Psi1)
            X2[:, i] = Normal(mean=m2[:, i], cov=self.Psi2)

        return X1.T, X2.T

    def _em_step(self, X, Lambda, Psi, n, k):
        """
        Perform Expectation-Maximization on the parameters Lambda and Psi. See
        my blog post on factor analysis:

        http://gregorygundersen.com/blog/2018/08/08/factor-analysis/

        Or the papers referenced in the docstring for derivations.

        Psi   : (p1 + p2) x (p1 + p2)
        Lambda: (p1 + p2) x k
        V     : (p1 + p2) x n
        """
        # Update Lambda.
        # ==============
        Exp          = self.E_z_given_x(Lambda, Psi, X)
        Lambda_lterm = dot(X, Exp.T)
        Lambda_rterm = self.E_zzT_given_x(Lambda, Psi, X, k)
        Lambda_star  = dot(Lambda_lterm, inv(Lambda_rterm))

        # Update Psi.
        # ===========
        Exp      = self.E_z_given_x(Lambda, Psi, X)
        Psi_new  = dot(X, X.T) - dot(Lambda_star, dot(Exp, X.T))
        Psi_star = 1./n * np.diag(np.diag(Psi_new))

        return Lambda_star, Psi_star

    def _init_params(self, p1, p2):
        """
        :param p1: Dimensionality of the first view of data.
        :param p2: Dimensionality of the second view of data.
        :return:
        """
        k = self.n_components

        # TODO: How to choose these?
        Sigma_init = 0.5
        Sigma1     = Sigma_init
        Sigma2     = Sigma_init

        self.Lambda1 = np.random.random((p1, k))
        self.Lambda2 = np.random.random((p2, k))

        Psi1_init = Sigma1 * np.eye(p1)
        Psi2_init = Sigma2 * np.eye(p2)

        Lambda = np.concatenate((self.Lambda1, self.Lambda2), axis=0)
        Psi    = np.block([[Psi1_init, np.zeros((p1, p2))],
                           [np.zeros((p2, p1)), Psi2_init]])

        return Lambda, Psi

    def neg_log_likelihood(self, X, Lambda, Psi):
        """Compute negative log-likelihood.

        For a derivation of the log-likelihood Q, see Appendix B in:

            The EM Algorithm for Mixtures of Factor Analyzers
            Ghahramani and Hinton
            http://mlg.eng.cam.ac.uk/zoubin/papers/tr-96-1.pdf
        """
        p, n = X.shape
        k = self.n_components
        Q = 0

        Ez  = self.E_z_given_x(Lambda, Psi, X).T
        Ezz = self.E_zzT_given_x(Lambda, Psi, X, k).T
        assert Ezz.shape == (k, k)
        A   = np.diag(mm(mm(X.T, inv(Psi)), X))
        B   = -2 * np.diag(mm(mm(mm(X.T, inv(Psi)), Lambda), Ez.T))
        C   = tr(mm(mm(mm(Lambda.T, inv(Psi)), Lambda), Ezz))
        Q  += (A + B).sum() + C

        D = -n / 2. * log(det(Psi))
        Q += D

        neg_Q = -Q  # Code clarity: don't miss that negative sign.
        return neg_Q

    def neg_log_likelihood_archive(self, X, Lambda, Psi):
        """Compute negative log-likelihood.

        For a derivation of the log-likelihood Q, see Appendix B in:

            The EM Algorithm for Mixtures of Factor Analyzers
            Ghahramani and Hinton
            http://mlg.eng.cam.ac.uk/zoubin/papers/tr-96-1.pdf
        """
        p, n = X.shape
        k = self.n_components
        Q = 0

        for xi in X.T:
            Ez  = self.E_z_given_x(Lambda, Psi, xi)
            Ezz = self.E_zzT_given_x(Lambda, Psi, xi, k)
            A = mm(mm(xi.T, inv(Psi)), xi)
            B = -2 * (mm(mm(mm(xi.T, inv(Psi)), Lambda), Ez))
            C = tr(mm(mm(mm(Lambda.T, inv(Psi)), Lambda), Ezz))
            Q += A + B + C

        D = -n / 2. * log(det(Psi))
        Q += D

        neg_Q = -Q  # Code clarity: don't miss that negative sign.
        return neg_Q

    def E_z_given_x(self, L, P, X):
        beta = mm(L.T, inv(mm(L, L.T) + P))
        return mm(beta, X)

    def E_zzT_given_x(self, L, P, X, k):
        beta = mm(L.T, inv(mm(L, L.T) + P))
        bX   = mm(beta, X)
        if len(X.shape) == 2:
            # See here for details:
            # https://stackoverflow.com/questions/48498662/
            _, N = X.shape
            bXXb = np.einsum('ib,ob->io', bX, bX)
            return N * (np.eye(k) - mm(beta, L)) + bXXb
        else:
            bXXb = np.outer(bX, bX.T)
            return np.eye(k) - mm(beta, L) + bXXb


# -----------------------------------------------------------------------------
# Example.
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
from   _datasets import load_lowrankcov


k = 10
X1, X2 = load_lowrankcov(N=200, P=50, Q=50, k=k)
pcca = PCCA(n_components=2)
pcca.fit([X1, X2], n_iters=100)
X1_, X2_ = pcca.sample(200)

fig, ax = plt.subplots()
ax.scatter(X1[:, 0], X1[:, 1], c='red', marker='.')
ax.scatter(X2[:, 0], X2[:, 1], c='blue', marker='.')
ax.scatter(X1_[:, 0], X1_[:, 1], c='orange', marker='*')
ax.scatter(X2_[:, 0], X2_[:, 1], c='cyan', marker='*')

plt.savefig('_figures/pcca.png')
