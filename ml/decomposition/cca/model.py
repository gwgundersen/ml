"""=============================================================================
Canonical correlation analysis. See:

    A Tutorial on Canonical Correlation Methods
    https://arxiv.org/pdf/1711.02391.pdf

    See:
    https://stats.stackexchange.com/questions/77287/
============================================================================="""

import numpy as np
import scipy

from   ml.model import Model
from   ml import linalg as LA

# ------------------------------------------------------------------------------

inv  = np.linalg.inv
sqrt = scipy.linalg.sqrtm
mm   = np.matmul

# ------------------------------------------------------------------------------

class CCA(Model):

    def __init__(self, n_components=2, use_svd=False):
        self.n_components = n_components
        self.use_svd      = use_svd

# ------------------------------------------------------------------------------

    def fit(self, Xa, Xb):
        """Fits CCA model parameters using default or user-specified method.

        :param Xa: Observations with shape (p_dim, n_samps).
        :param Xb: Observations with shape (q_dim, n_samps).
        :return:   Class instance.
        """
        # Uurtio: "Throughout this tutorial, we assume that the variables are
        # standardised to zero mean and unit variance.
        Xa -= Xa.mean(axis=0)
        Xa /= Xa.std(axis=0)
        Xb -= Xb.mean(axis=0)
        Xb /= Xb.std(axis=0)

        p = Xa.shape[1]
        C   = LA.cov(Xa, Xb)
        Caa = C[:p, :p]
        Cbb = C[p:, p:]
        Cab = C[:p, p:]
        Cba = C[p:, :p]

        if self.use_svd:
            Wa, Wb = self._fit_svd(Xa, Xb, Caa, Cab, Cbb)
        else:
            Wa, Wb = self._fit_standard_eigval_prob(Xa, Xb, Caa, Cab, Cba, Cbb)

        self.Xa = Xa
        self.Xb = Xb

        self.Wa = Wa
        self.Wb = Wb

        return self

# ------------------------------------------------------------------------------

    def transform(self, Xa=None, Xb=None):
        """
        :param Xa:
        :param Xb:
        :return:
        """
        if Xa is None:
            Xa = self.Xa
        if Xb is None:
            Xb = self.Xb
        # Ensure embeddings are on the surface of a unit ball.
        Za = LA.norm_columns(mm(Xa, self.Wa))
        Zb = LA.norm_columns(mm(Xb, self.Wb))
        return Za[:, :self.n_components], Zb[:, :self.n_components]

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

    def _fit_standard_eigval_prob(self, Xa, Xb, Caa, Cab, Cba, Cbb):
        """Fits CCA model parameters using the standard eigenvalue problem.

        :param Xa: Observations with shape (p_dim, n_samps).
        :param Xb: Observations with shape (q_dim, n_samps).
        :return:   None.
        """
        N, p = Xa.shape
        N, q = Xb.shape
        r    = min(LA.rank(Xa), LA.rank(Xb))

        # Either branch results in: r x r matrix where r = min(p, q).
        if q < p:
            M = mm(mm(inv(Cbb), Cba), mm(inv(Caa), Cab))
        else:
            M = mm(mm(inv(Caa), Cab), mm(inv(Cbb), Cba))
        assert M.shape == (r, r)

        # Solving the characteristic equation,
        #
        #     det(M - rho^2 I) = 0
        #
        # is equivalent to solving for rho^2, which are the eigenvalues of the
        # matrix.
        eigvals, eigvecs = LA.eig(M)
        rhos = np.sqrt(eigvals)

        # Ensure we go through eigenvectors in descending order.
        inds    = (-rhos).argsort()
        rhos    = rhos[inds]
        eigvals = eigvals[inds]
        # NumPy returns each eigenvector as a column in a matrix.
        eigvecs = eigvecs.T[inds].T
        Wb      = eigvecs

        # This is just a sanity check to verify that eigenvalues and
        # eigenvectors are being computed correctly.
        for val, vec in zip(eigvals, eigvecs.T):
            M_ = M - (np.eye(r) * val)
            assert np.allclose(mm(M_, vec), 0)

        Wa = np.zeros((p, r))
        for i, (rho, wb_i) in enumerate(zip(rhos, Wb.T)):
            wa_i = mm(mm(inv(Caa), Cab), wb_i) / rho
            Wa[:, i] = wa_i

        # Sanity check: canonical correlations are equal to the rhos.
        Za = LA.norm_columns(mm(Xa, Wa))
        Zb = LA.norm_columns(mm(Xb, Wb))
        CCs = np.zeros(r)
        for i in range(r):
            za = Za[:, i]
            zb = Zb[:, i]
            CCs[i] = np.dot(za, zb)
        assert np.allclose(CCs, rhos)

        return Wa, Wb

# ------------------------------------------------------------------------------

    def _fit_svd(self, Xa, Xb, Caa, Cab, Cbb):
        """Fits CCA model parameters using SVD.

        :param Xa: Observations with shape (p_dim, n_samps).
        :param Xb: Observations with shape (q_dim, n_samps).
        :return:   None.
        """
        N, p = Xa.shape
        N, q = Xb.shape
        r    = min(LA.rank(Xa), LA.rank(Xb))

        Caa_sqrt = sqrt(Caa)
        Cbb_sqrt = sqrt(Cbb)

        Caa_sqrt_inv = inv(Caa_sqrt)
        Cbb_sqrt_inv = inv(Cbb_sqrt)

        # See Uurtio, eq. 12.
        A = mm(mm(Caa_sqrt_inv, Cab), Cbb_sqrt_inv)

        U, S, V = LA.svd(A, full_matrices=False)
        assert np.allclose(A, np.dot(U[:, :A.shape[1]] * S, V))

        # See Uurtio, eq. after eq. 12 (not numbered).
        Wa = mm(Caa_sqrt_inv, U)
        Wb = mm(Cbb_sqrt_inv, V.T)

        # Sanity check: the singular values of the matrix S to correspond to the
        # canonical correlations.
        Za = LA.norm_columns(mm(Xa, Wa))
        Zb = LA.norm_columns(mm(Xb, Wb))
        for s, za, zb in zip(S, Za.T, Zb.T):
            assert np.isclose(np.dot(za, zb), s)

        return Wa, Wb
