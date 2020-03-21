"""=============================================================================
Interface for old faithful eruptions dataset.
============================================================================="""

from   .oldfaithful.oldfaithful import load as load_oldfaithful
from   .iris import load as load_iris
from   .lowrankcov import load as load_lowrankcov
from   .paired import load as load_paired
from   .synthetic import load as load_synthetic