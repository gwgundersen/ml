"""=============================================================================
Model base class.
============================================================================="""

class Model(object):

    def fit(self):
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def transform(self):
        raise NotImplementedError()

# ------------------------------------------------------------------------------

    def fit_transform(self):
        raise NotImplementedError()
