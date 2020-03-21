"""============================================================================
Factor analysis.
============================================================================"""

import numpy as np


# -----------------------------------------------------------------------------

class FA:

    def __init__(self):
        self.E_z_given_v_i  = E_z_given_v_i_Bishop
        self.E_zzT_give_v_i = E_zzT_give_v_i_Bishop

    def fit(self, n_iters=100):

        self.Lambda, self.Psi = self._init_params()

        for _ in range(n_iters):
            Lambda_new, Psi_new = self._em_step(self.Lambda, self.Psi, V)
            self.Lambda = Lambda_new
            self.Psi    = Psi_new

    def transform(self):
        pass

    def _init_params(self):
        """
        :return:
        """
        sigma_init = 0.5
        W_x_init   = np.random.random((p_x, k))
        W_y_init   = np.random.random((p_y, k))

        Psi_x_init = sigma_x * np.eye(p_x)
        Psi_y_init = sigma_y * np.eye(p_y)

        Lambda_old = np.concatenate((W_x_init, W_y_init), axis=0)
        Psi_old    = np.block([[Psi_x_init, np.zeros((p_x, p_y))],
                               [np.ones((p_y, p_x)), Psi_y_init]])

        return Lambda_old, Psi_old

    def _em_step(self, Lambda, Psi, V):
        """
        Psi   : (p_x + p_y) by (p_x + p_y)
        Lambda: (p_x + p_y) by k
        V     : (p_x + p_y) by n
        """
        n, p = V.shape
        p, k = Lambda.shape

        # update lambda
        # -------------

        # These are the two terms in the Lambda update
        Lambda_new_1 = np.zeros((p, k))
        Lambda_new_2 = np.zeros((k, k))

        for i in range(n):

            # Expectation terms
            # The difference between Bishop and Murphy is that
            # Bishop's derivation uses the Woodbury identity (see G&H)
            # while Murphy implementation just uses Numpy's built-in inverse
            # function.
            Exp_i        = E_z_given_v_i_Bishop(Lambda, Psi, V[:,i,None])
            Cov_i        = E_zzT_give_v_i_Bishop(Lambda, Psi, V[:,i,None])

            Lambda_new_1 += np.dot(V[:,i,None], Exp_i.T)
            Lambda_new_2 += Cov_i

        Lambda_star  = np.dot(Lambda_new_1, np.linalg.inv(Lambda_new_2))

        # update psi
        # ----------
        Psi_new      = np.zeros(Psi.shape)
        for i in range(n):
            Exp_i = E_z_given_v_i_Bishop(Lambda, Psi, V[:,i,None])
            A = Psi_new + np.dot(V[:, i, None], V[:, i, None].T)
            B = np.dot(Lambda_star, np.dot(Exp_i, V[:, i, None].T))
            Psi_new = A - B

        Psi_star     = 1./n * np.diag(np.diag(Psi_new))

        return Lambda_star, Psi_star


# -----------------------------------------------------------------------------
# Update functions.
# -----------------------------------------------------------------------------

def E_z_given_v_i_Bishop(Lambda, Psi, vi):
    """
    :param Lambda:
    :param Psi:
    :param vi:
    :return:
    """
    # 12.66, 12.67, 12.68
    LT_P_L = np.dot(Lambda.T, np.dot(np.linalg.inv(Psi), Lambda))
    G      = np.linalg.inv(np.eye(LT_P_L.shape[0]) + LT_P_L)

    beta   = np.dot(G, np.dot(Lambda.T, np.linalg.inv(Psi)))

    return np.dot(beta, vi)


def E_zzT_give_v_i_Bishop(Lambda, Psi, vi):
    """
    :param Lambda:
    :param Psi:
    :param vi:
    :return:
    """
    LT_P_L = np.dot(Lambda.T, np.dot(np.linalg.inv(Psi), Lambda))
    G      = np.linalg.inv(np.eye(LT_P_L.shape[0]) + LT_P_L)

    E_z    =  E_z_given_v_i_Bishop(Lambda, Psi, vi)

    return G + np.dot(E_z, E_z.T)
