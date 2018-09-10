"""=============================================================================
Probabilistic canonical correlation analysis. For references in comments:

    A Probabilistic Interpretation of Canonical Correlation Analysis.
    Bach, Jordan (2006).

    The EM algorithm for mixtures of factor analyzers.
    Ghahramani, Hinton (1996).

============================================================================="""

import numpy as np
from   ml.model import Model

# ------------------------------------------------------------------------------

inv    = np.linalg.inv
dot    = np.dot
Normal = np.random.multivariate_normal

# ------------------------------------------------------------------------------

class PCCA(Model):

    def __init__(self, n_components, derivation='Bishop'):
        """
        :param n_components:
        :param derivation:
        :return:
        """
        self.n_components = n_components

        # The difference between Bishop and Murphy is that Bishop's derivation
        # uses the Woodbury identity (see Ghahramani 1996), while Murphy's
        # derivation does not specify how to compute the inverse. We implement
        # this using NumPy's inverse function, `np.linalg.inv`.
        DERIVATIONS = ['Bishop', 'Murphy']

        if derivation not in DERIVATIONS:
            msg = 'Supported derivations: %s' % ', '.join(DERIVATIONS)
            raise AttributeError(msg)

        if derivation == 'Bishop':
            self.E_z_given_x   = _E_z_given_x_Bishop
            self.E_zzT_given_x = _E_zzT_given_x_Bishop
        else:
            self.E_z_given_x   = _E_z_given_x_Murphy
            self.E_zzT_given_x = _E_zzT_given_x_Murphy

# ------------------------------------------------------------------------------

    def fit(self, X, n_iters=100):
        """
        :param X:
        :param n_iters:
        :return:
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

        Lambda, Psi = self._init_params(p1, p2)
        for i in range(n_iters):
            print('Iter: %s' % i)
            Lambda_new, Psi_new = self._em_step(X, Lambda, Psi, n, p, k)
            Lambda = Lambda_new
            Psi    = Psi_new

        self.X       = X
        self.Lambda  = Lambda
        self.Lambda1 = Lambda[:p1, :]
        self.Lambda2 = Lambda[p1:, :]
        self.Psi     = Psi
        self.Psi1    = Psi[:p1, :p1]
        self.Psi2    = Psi[p1:, p1:]

# ------------------------------------------------------------------------------

    def transform(self):
        n = self.X.shape[0]
        Z = np.zeros((self.n_components, n))
        for i in range(n):
            Z[:, i] = self.E_z_given_x(self.Lambda, self.Psi, self.X[:, i])
        return Z

# ------------------------------------------------------------------------------

    def sample(self, n):
        Z  = np.random.random((self.n_components, n))

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

    def _init_params(self, p1, p2):
        """
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

    def _em_step(self, X, Lambda, Psi, n, p, k):
        """
        Psi   : (p1 + p2) x (p1 + p2)
        Lambda: (p1 + p2) x k
        V     : (p1 + p2) x n
        """
        # Update Lambda.
        # ======================================================================
        Exp          = self.E_z_given_x(Lambda, Psi, X)
        Lambda_lterm = dot(X, Exp.T)
        Lambda_rterm = self.E_zzT_given_x(Lambda, Psi, X, k)
        Lambda_star  = dot(Lambda_lterm, inv(Lambda_rterm))

        # Update Psi.
        # ======================================================================
        Exp      = self.E_z_given_x(Lambda, Psi, X)
        Psi_new  = dot(X, X.T) - dot(Lambda_star, dot(Exp, X.T))
        Psi_star = 1./n * np.diag(np.diag(Psi_new))

        return Lambda_star, Psi_star

# ------------------------------------------------------------------------------

def _E_z_given_x_Bishop(Lambda, Psi, x):
    """
    :param Lambda:
    :param Psi:
    :param x:
    :return:
    """
    # 12.66, 12.67, 12.68
    LT_P_L = np.dot(Lambda.T, np.dot(np.linalg.inv(Psi), Lambda))
    G      = np.linalg.inv(np.eye(LT_P_L.shape[0]) + LT_P_L)
    beta   = np.dot(G, np.dot(Lambda.T, np.linalg.inv(Psi)))
    return np.dot(beta, x)

# ------------------------------------------------------------------------------

def _E_zzT_given_x_Bishop(Lambda, Psi, x, _):
    """
    :param Lambda:
    :param Psi:
    :param x:
    :return:
    """
    LT_P_L = np.dot(Lambda.T, np.dot(np.linalg.inv(Psi), Lambda))
    G      = np.linalg.inv(np.eye(LT_P_L.shape[0]) + LT_P_L)
    E_z    =  _E_z_given_x_Bishop(Lambda, Psi, x)
    return G + np.dot(E_z, E_z.T)

# ------------------------------------------------------------------------------

def _E_z_given_x_Murphy(L, P, x):
    """
    :param L:
    :param P:
    :param x:
    :return:
    """
    beta = dot(L.T, inv(dot(L, L.T) + P))
    return dot(beta, x)

# ------------------------------------------------------------------------------

def _E_zzT_given_x_Murphy(L, P, x, k):
    """
    :param L:
    :param P:
    :param x:
    :param k:
    :return:
    """
    beta = dot(L.T, inv(dot(L, L.T) + P))
    return np.eye(k) - dot(beta, L) + dot(dot(beta, x), dot(x.T, beta.T))
