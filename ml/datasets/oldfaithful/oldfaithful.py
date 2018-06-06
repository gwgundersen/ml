"""=============================================================================
Old faithful eruptions dataset.
============================================================================="""

import numpy as np

# ------------------------------------------------------------------------------

def load():
    """
    :return: Old faithful reuprtions data with shape (272, 2).
    """
    return np.loadtxt('ml/datasets/oldfaithful/data', skiprows=1)[:, 1:]
