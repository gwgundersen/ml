"""=============================================================================
Probabilistic canonical correlation analysis. For references in comments:

    A Probabilistic Interpretation of Canonical Correlation Analysis
    Bach, Jordan
    https://www.di.ens.fr/~fbach/probacca.pdf
============================================================================="""

import numpy as np

from   ml.model import Model
from   ml import linalg as LA
from   ml.linalg import inv, mm, sqrt

# ------------------------------------------------------------------------------

spectral_norm = lambda X: np.linalg.norm(X, ord=2)

# ------------------------------------------------------------------------------

class PCCA(Model):

    def __init__(self, n_components=2):
        self.n_components = n_components

# ------------------------------------------------------------------------------

    def fit(self, X1, X2, n_iters=100):
        """Fits probabilistic CCA model parameters, W1, W2, Ψ1, Ψ2, μ1, μ2.

        :param X1: Observations with shape (p_dim, n_samps).
        :param X2: Observations with shape (q_dim, n_samps).
        :return:   Class instance.
        """
        N, m1 = X1.shape
        N, m2 = X2.shape
        m     = min(m1, m2)

        P, U1, U2, CCs = self.ccs(X1, X2)

        C   = LA.cov(X1, X2)
        C11 = C[:m1, :m1]
        C22 = C[m1:, m1:]
        C12 = C[:m1, m1:]
        C21 = C[m1:, :m1]

        M1 = M2 = LA.cholesky(P)
        # "The spectral norms of M_1 and M_2 are smaller than one."
        assert spectral_norm(M1) < 1
        assert np.allclose(mm(M1, M1.T), P)

        for i in range(n_iters):
            self.e_step(C11, C22, C12, C21, U1, U2, M1, M2)
            # self.m_step()
        print('fin')

# ------------------------------------------------------------------------------

    def e_step(self, C11, C22, C12, C21, U1, U2, M1, M2):
        self.W1   = mm(mm(C11, U1), M1)
        self.W2   = mm(mm(C22, U2), M2)
        self.Psi1 = C11 - mm(self.W1, self.W1.T)
        self.Psi2 = C22 - mm(self.W2, self.W2.T)
        # self.mu1  = mu1
        # self.mu2  = mu2

# ------------------------------------------------------------------------------

    def transform(self, Xa=None, Xb=None):
        """
        :param Xa: Observations with shape (p_dim, n_samps).
        :param Xb: Observations with shape (q_dim, n_samps).
        :return:   Embeddings for each data view.
        """
        pass

# ------------------------------------------------------------------------------

    def fit_transform(self, Xa, Xb):
        """
        Fit model and then transforms data using learned model.

        :param Xa: Observations with shape (p_dim, n_samps).
        :param Xb: Observations with shape (q_dim, n_samps).
        :return:   Result of self.transform() function.
        """
        self.fit(Xa, Xb)
        return self.transform()

# ------------------------------------------------------------------------------

    def ccs(self, X1, X2):
        N, m1 = X1.shape
        N, m2 = X2.shape
        m     = min(m1, m2)

        C   = LA.cov(X1, X2)
        C11 = C[:m1, :m1]
        C22 = C[m1:, m1:]
        C12 = C[:m1, m1:]
        C21 = C[m1:, :m1]

        M = mm(mm(sqrt(inv(C11)), C12), sqrt(inv(C22)))
        assert M.shape == (m1, m2)

        # NumPy documentation:
        #
        #     U:  matrix having left singular vectors as columns.
        #     Vh: matrix having right singular vectors as rows.
        #
        U, S, Vh = LA.svd(M)

        # Left singular vectors as column vectors.
        V1 = U.T[:m].T
        # Right singular vectors as column vectors.
        V2 = Vh[:m].T

        U1 = mm(sqrt(inv(C11)), V1)
        U2 = mm(sqrt(inv(C22)), V2)
        assert U1.shape == (m1, m)
        assert U2.shape == (m2, m)

        assert np.allclose(mm(mm(U1.T, C11), U1), np.eye(m))
        assert np.allclose(mm(mm(U2.T, C22), U2), np.eye(m))

        P = mm(mm(U2.T, C21), U1)
        # P is an (q, m)-diagonal matrix.
        P_are0s = np.isclose(P, 0)
        I_are0s = np.eye(m) == 0
        assert np.allclose(P_are0s, I_are0s)

        # Canonical correlations are on the diagonal of P.
        CCs = np.diag(P)

        return P, U1, U2, CCs