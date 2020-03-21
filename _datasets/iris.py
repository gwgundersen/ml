"""=============================================================================
The Iris dataset.
============================================================================="""

import numpy as np
from   sklearn import datasets

# ------------------------------------------------------------------------------

def load():
    iris = datasets.load_iris()
    x    = np.transpose(iris.data)
    y    = iris.target
    return x, y
