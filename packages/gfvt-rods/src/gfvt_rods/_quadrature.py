"""Shared quadrature for the polymer reservoir integrals."""

from numpy.polynomial.legendre import leggauss


# phi_R = upper*t**4 smooths the fractional powers in the PEG equation of
# state at the origin. This avoids thousands of uniform quadrature nodes.
_NODES, _WEIGHTS = leggauss(96)
_T = (_NODES + 1) / 2
_W = _WEIGHTS / 2
