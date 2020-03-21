"""============================================================================
Canonical correlation analysis. See paper below for comment references:

    A Tutorial on Canonical Correlation Methods
    https://arxiv.org/pdf/1711.02391.pdf
============================================================================"""

import _linalg as LA

mm   = LA.mm
inv  = LA.inv
sqrt = LA.sqrt


# -----------------------------------------------------------------------------

class CCA:

    def __init__(self, n_components=2, use_svd=False):
        self.n_components = n_components
        self.use_svd      = use_svd

    def fit(self, Xa, Xb):
        """Fits CCA model parameters using default or user-specified method.

        :param Xa: Observations with shape (n_samps, p_dim).
        :param Xb: Observations with shape (n_samps, q_dim).
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

    def transform(self, Xa=None, Xb=None):
        """
        :param Xa: Observations with shape (n_samps, p_dim).
        :param Xb: Observations with shape (n_samps, q_dim).
        :return:   Embeddings for each data view.
        """
        if Xa is None:
            Xa = self.Xa
        if Xb is None:
            Xb = self.Xb
        # Ensure embeddings are on the surface of a unit ball.
        Za = LA.norm_columns(mm(Xa, self.Wa))
        Zb = LA.norm_columns(mm(Xb, self.Wb))
        return Za[:, :self.n_components], Zb[:, :self.n_components]

    def fit_transform(self, Xa, Xb):
        """
        Fit model and then transforms data using learned model.

        :param Xa: Observations with shape (n_samps, p_dim).
        :param Xb: Observations with shape (n_samps, q_dim).
        :return:   Result of self.transform() function.
        """
        self.fit(Xa, Xb)
        return self.transform()

    def _fit_standard_eigval_prob(self, Xa, Xb, Caa, Cab, Cba, Cbb):
        """Fits CCA model parameters using the standard eigenvalue problem.

        :param Xa: Observations with shape (n_samps, p_dim).
        :param Xb: Observations with shape (n_samps, q_dim).
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
        BJCCs = self._bach_jordan_ccs(Xa, Xb)
        for i in range(r):
            za = Za[:, i]
            zb = Zb[:, i]
            CCs[i] = np.dot(za, zb)
        # assert np.allclose(CCs, rhos)
        # assert np.allclose(CCs, BJCCs)

        return Wa, Wb

    def _fit_svd(self, Xa, Xb, Caa, Cab, Cbb):
        """Fits CCA model parameters using SVD.

        :param Xa: Observations with shape (n_samps, p_dim).
        :param Xb: Observations with shape (n_samps, q_dim).
        :return:   None.
        """
        N, p = Xa.shape
        N, q = Xb.shape
        r    = min(LA.rank(Xa), LA.rank(Xb))

        Caa_sqrt = LA.sqrt(Caa)
        Cbb_sqrt = LA.sqrt(Cbb)

        Caa_sqrt_inv = inv(Caa_sqrt)
        Cbb_sqrt_inv = inv(Cbb_sqrt)

        # See Uurtio, eq. 12.
        A = mm(mm(Caa_sqrt_inv, Cab), Cbb_sqrt_inv)

        U, S, V = LA.svd(A, full_matrices=False)
        assert np.allclose(A, np.dot(U[:, :A.shape[1]] * S, V))

        # See Uurtio, eq. after eq. 12 (not numbered).
        Wa = mm(Caa_sqrt_inv, U)
        Wb = mm(Cbb_sqrt_inv, V.T)

        # Sanity check: the singular values of the matrix S to correspond to
        # the canonical correlations.
        Za = LA.norm_columns(mm(Xa, Wa))
        Zb = LA.norm_columns(mm(Xb, Wb))
        BJCCs = self._bach_jordan_ccs(Xa, Xb)
        for s, bj, za, zb in zip(S, BJCCs, Za.T, Zb.T):
            assert np.isclose(s,  np.dot(za, zb))
            assert np.isclose(bj, np.dot(za, zb))

        return Wa, Wb

    def _bach_jordan_ccs(self, X1, X2):
        """Returns the canonical correlations according to Bach and Jordan
        (2006). Used for sanity checking.

        :param X1: Observations with shape (n_samps, p_dim).
        :param X2: Observations with shape (n_samps, q_dim).
        :return:   Class instance.
        """
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

        return CCs


# -----------------------------------------------------------------------------
# Example.
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    N = 50
    P = 5
    amean = np.zeros(P)
    acov = np.eye(P)
    Xa = np.random.multivariate_normal(mean=amean, cov=acov, size=N)
    Xb = Xa * 5

    Xa = Xa - Xa.mean(axis=0)
    Xb = Xb - Xb.mean(axis=0)

    cca = CCA(n_components=2, use_svd=False)
    Z1, Z2 = cca.fit_transform(Xa, Xb)

    fig, ax = plt.subplots()
    ax.scatter(Z1[:, 0], Z1[:, 1], c='blue')
    ax.scatter(Z2[:, 0], Z2[:, 1], c='red')

    for i, txt in enumerate(range(len(Z1))):
        ax.annotate(txt, (Z1[:, 0][i], Z1[:, 1][i]), color='blue')

    for i, txt in enumerate(range(len(Z2))):
        ax.annotate(txt, (Z2[:, 0][i], Z2[:, 1][i]), color='red')

    plt.show()
