"""Compatibility alias for the canonical generalized PINN library.

Older scripts imported ``pinn_lib_kepler_opti`` and mutated module-level
configuration flags. Replacing the module entry (instead of using ``import *``)
keeps those assignments connected to the canonical implementation.
"""

import sys

import pinn_lib_kepler as _canonical


sys.modules[__name__] = _canonical
