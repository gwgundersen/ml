"""=============================================================================
Probabilistic canonical correlation analysis. For references in comments:

    A Probabilistic Interpretation of Canonical Correlation Analysis.
    Bach, Jordan (2006).

    The EM algorithm for mixtures of factor analyzers.
    Ghahramani, Hinton (1996).
============================================================================="""

import numpy as np
from   ml.model import Model
from   ml import linalg as LA

# ------------------------------------------------------------------------------

mm     = np.matmul
diag   = np.diag
det    = np.linalg.det
dot    = np.dot
log    = np.log
tr     = np.trace
inv    = np.linalg.inv
Normal = np.random.multivariate_normal

# ------------------------------------------------------------------------------

class RPCCA(Model):

    def __init__(self, n_components, rank_k=None):
        """
        Initialize the probabilistic CCA model.

        :param n_components: The dimensionality of the latent variable.
        :param rank_k:       The rank to use for low-rank approximation of matrix
                             inversion.
        :return:             None.
        """
        self.n_components = n_components
        self.rank_k = rank_k

# ------------------------------------------------------------------------------

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
            Lambda_new, Psi_new = self._em_step(X, Lambda, Psi, n, k)
            nll = self.neg_log_likelihood(X, Lambda, Psi)
            print('%s: %s' % (i, nll))
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

# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------

    def _em_step(self, X, Lambda, Psi, n, k):
        """
        Perform Expectation–Maximization on the parameters Lambda and Psi. See
        my blog post on factor analysis:

        http://gregorygundersen.com/blog/2018/08/08/factor-analysis/

        Or the papers referenced in the docstring for derivations.

        Psi   : (p1 + p2) x (p1 + p2)
        Lambda: (p1 + p2) x k
        V     : (p1 + p2) x n
        """
        # Update Lambda.
        # ======================================================================
        Exp          = self.E_z_given_x(Lambda, Psi, X)
        Lambda_lterm = dot(X, Exp.T)
        Lambda_rterm = self.E_zzT_given_x(Lambda, Psi, X, k)
        Lambda_star  = dot(Lambda_lterm, self.inv(Lambda_rterm))

        # Update Psi.
        # ======================================================================
        Exp      = self.E_z_given_x(Lambda, Psi, X)
        Psi_new  = dot(X, X.T) - dot(Lambda_star, dot(Exp, X.T))
        Psi_star = 1./n * np.diag(np.diag(Psi_new))

        return Lambda_star, Psi_star

# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------

    def E_z_given_x(self, L, P, X):
        beta  = mm(L.T, self.inv(mm(L, L.T) + P))
        return mm(beta, X)

# ------------------------------------------------------------------------------

    def E_zzT_given_x(self, L, P, X, k):
        N = X.shape[0]
        beta = mm(L.T, self.inv(mm(L, L.T) + P))
        return N * (np.eye(k) - mm(beta, L)) + mm(mm(beta, X), mm(X.T, beta.T))

# ------------------------------------------------------------------------------

    def inv(self, X):
        if self.rank_k:
            return LA.rinv(X, self.rank_k)
        return np.linalg.inv(X)