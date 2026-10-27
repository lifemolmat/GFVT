"""Smooth density-dependent PEG exclusion and stable fluid coexistence.

For x=phi-phi_c > 0 use H_w(x)=x*exp(-w/x), zero otherwise. All derivatives
vanish at x=0. H'=exp(-w/x)*(1+w/x) increases from zero to one without
overshooting. The ramp requires no quadrature or special functions. This is
a phenomenological exclusion law, not a microscopic hydration model.
"""

from math import comb, factorial

import numpy as np
from scipy.optimize import brentq, least_squares

from .common_tangent import _DenseMinimum, _coexistence
from .exclusion import ExclusionLayerEOS, _ExclusionReservoir
from .rods import _real_scalar


def smooth_ramp(x, width, order=4):
    """H_w and ordinary derivatives through order 4; x and width are fractions.

    H_w(x)=x*exp(-width/x) for x>0 and zero otherwise. Evaluate only the
    positive side to avoid division by zero or exponential overflow at the
    join. width is strictly positive; sharp mode is separate.
    """
    width = _real_scalar(width, "width", positive=True)
    if order not in range(5):
        raise ValueError("order must be between 0 and 4.")
    x = np.asarray(x, dtype=float)
    result = [np.zeros_like(x) for _ in range(order + 1)]
    # Beyond t=745 exp(-t) underflows. Mask before taking reciprocals so
    # arbitrarily small positive x still approaches the flat joining point.
    mask = x > width / 745
    t = width / x[mask]
    exp = np.exp(-t)
    result[0][mask] = x[mask] * exp
    if order >= 1:
        result[1][mask] = (1 + t) * exp
    if order >= 2:
        result[2][mask] = t**3 * exp / width
    if order >= 3:
        result[3][mask] = t**4 * (t - 3) * exp / width**2
    if order >= 4:
        result[4][mask] = t**5 * (t**2 - 8 * t + 12) * exp / width**3
    return np.asarray(result)


class SmoothExclusionLayerEOS(ExclusionLayerEOS):
    def __init__(self, q, solvent, s, phi_c, mu_constant=0, *, width=0.1, nu=None):
        super().__init__(q, solvent, s, phi_c, mu_constant, nu=nu)
        self.width = _real_scalar(width, "width", positive=True)

    def layer_increment(self, phi):
        return self.s * smooth_ramp(np.asarray(phi) - self.phi_c, self.width, order=1)

    def at(self, phi_R):
        return _SmoothExclusionReservoir(self, phi_R)


def _polynomial_derivatives(x, coefficients):
    """Derivatives of a cubic polynomial composed with a scalar function x."""
    c0, c1, c2, c3 = coefficients
    p = [c0 + x[0] * (c1 + x[0] * (c2 + c3 * x[0]))]
    first = c1 + 2 * c2 * x[0] + 3 * c3 * x[0]**2
    second = 2 * c2 + 6 * c3 * x[0]
    third = 6 * c3
    if len(x) > 1:
        p.append(first * x[1])
    if len(x) > 2:
        p.append(first * x[2] + second * x[1]**2)
    if len(x) > 3:
        p.append(first * x[3] + 3 * second * x[1] * x[2] + third * x[1]**3)
    if len(x) > 4:
        p.append(first * x[4] + second * (4 * x[1] * x[3] + 3 * x[2]**2)
                 + 6 * third * x[1]**2 * x[2])
    return p


class _SmoothExclusionReservoir(_ExclusionReservoir):
    def alpha_derivatives(self, phi, qx=None, order=4):
        """Total derivatives of alpha at fixed reservoir, including qx(phi)."""
        if order not in range(5):
            raise ValueError("order must be between 0 and 4.")
        phi = np.asarray(phi)[..., None]
        qx = self.grid_qx if qx is None else np.asarray(qx)
        ramp = smooth_ramp(phi - self.eos.phi_c, self.eos.width, order=order)
        q = [self.eos.s * r for r in ramp]
        q[0] = q[0] + qx
        y = [phi / (1 - phi)] + [factorial(k) / (1 - phi)**(k + 1) for k in range(1, order + 1)]
        a = _polynomial_derivatives(q, (0, 3, 3, 1))
        b = _polynomial_derivatives(q, (0, 0, 4.5, 3))
        c = _polynomial_derivatives(q, (0, 0, 0, 3))
        y2 = _polynomial_derivatives(y, (0, 0, 1, 0))
        y3 = _polynomial_derivatives(y, (0, 0, 0, 1))
        Q = [sum(comb(n, k) * (a[k] * y[n - k] + b[k] * y2[n - k] + c[k] * y3[n - k])
                 for k in range(n + 1)) for n in range(order + 1)]
        alpha = (1 - phi) * np.exp(-Q[0])
        L = [None] + [-factorial(k - 1) / (1 - phi)**k - Q[k] for k in range(1, order + 1)]
        result = [alpha]
        if order >= 1:
            result.append(alpha * L[1])
        if order >= 2:
            result.append(alpha * (L[1]**2 + L[2]))
        if order >= 3:
            result.append(alpha * (L[1]**3 + 3 * L[1] * L[2] + L[3]))
        if order >= 4:
            result.append(alpha * (L[1]**4 + 6 * L[1]**2 * L[2] + 3 * L[2]**2
                                    + 4 * L[1] * L[3] + L[4]))
        return np.asarray(result)

    @staticmethod
    def _reference_pressure_derivatives(phi):
        return np.array([
            (1 + 4 * phi + 4 * phi**2 - 4 * phi**3 + phi**4) / (1 - phi)**4,
            4 * (2 + 5 * phi - phi**2) / (1 - phi)**5,
            12 * (5 + 6 * phi - phi**2) / (1 - phi)**6,
        ])

    def slopes(self, phi):
        """dPi/dphi and d2Pi/dphi2, with respect to physical volume fraction."""
        phi = np.asarray(phi)
        a = self.alpha_derivatives(phi, order=3)
        a2, a3 = (np.sum(self.weights * a[k], axis=-1) for k in (2, 3))
        return self._reference_pressure_derivatives(phi)[:2] - np.array([phi * a2, a2 + phi * a3])

    def free_derivatives(self, phi):
        """Omega, dOmega/dphi, ..., d4Omega/dphi4, at fixed phi_R.

        These include the complete chain rule for the density-dependent layer;
        the first two pressure derivatives follow from phi*Omega'' and
        Omega''+phi*Omega'''. The fourth derivative checks critical stability.
        """
        phi = np.asarray(phi)
        a = self.alpha_derivatives(phi)
        integrals = np.sum(self.weights * a, axis=-1)
        free, mu, _ = self.reference(phi)
        p1, p2, p3 = self._reference_pressure_derivatives(phi)
        reference = np.array([free, mu, p1 / phi, p2 / phi - p1 / phi**2,
                              p3 / phi - 2 * p2 / phi**2 + 2 * p1 / phi**3])
        return reference - integrals


def _local_pair(state, previous):
    """Spinodal-separated continuation in log density, excluding equal roots."""
    floor, ceiling, phi_c = np.finfo(float).tiny, 1 - 1e-8, state.eos.phi_c
    slope = lambda phi: state.slopes(phi)[0]
    spin_g = brentq(slope, floor, phi_c, xtol=1e-15)
    # Take the first dense spinodal, so this seed starts on the fluid branch
    # connected to criticality even when other dense minima exist.
    grid = np.unique(np.r_[phi_c, phi_c + np.geomspace(1e-12, ceiling - phi_c, 100)])
    slopes = state.slopes(grid)[0]
    positive = np.flatnonzero(slopes > 0)
    if not len(positive):
        raise RuntimeError("Could not locate the dense spinodal of the smooth exclusion model.")
    i = positive[0]
    if i == 0:
        raise RuntimeError("Reservoir concentration is numerically indistinguishable from criticality.")
    spin_l = brentq(slope, grid[i - 1], grid[i], xtol=1e-15)
    midpoint = (spin_g + spin_l) / 2
    half_gap = np.sqrt(3) * (spin_l - spin_g) / 2
    seed = [max(midpoint - half_gap, spin_g / 2), min(midpoint + half_gap, (1 + spin_l) / 2)]
    lower, upper = np.log([floor, spin_l]), np.log([spin_g, ceiling])

    def residual(log_phi):
        gas, liquid = np.exp(log_phi)
        return (state.evaluate(gas)[1:] - state.evaluate(liquid)[1:]) / (liquid - gas)

    for guess in (previous, seed):
        if guess is None:
            continue
        sol = least_squares(residual, np.clip(np.log(guess), lower + 1e-10, upper - 1e-10),
                            bounds=(lower, upper), x_scale="jac", gtol=1e-12,
                            ftol=1e-12, xtol=1e-12, max_nfev=150)
        if sol.success and not np.any(sol.active_mask) and np.max(np.abs(sol.fun)) < 1e-8:
            pair = np.exp(sol.x)
            if np.all(state.slopes(pair)[0] > 0):
                return pair
    raise RuntimeError(f"Could not resolve a smooth fluid pair at phi_R={state.phi_R:.10g}.")


def stable_pair(state, previous):
    """Return a globally supporting fluid pair; never accept a metastable root."""
    try:
        pair = _local_pair(state, previous)
    except (ValueError, RuntimeError):
        pair = None
    dense = _DenseMinimum(state)
    if pair is not None:
        _, mu, pressure = state.evaluate(pair[0])
        gap, _ = dense(mu, pressure)
        if gap >= -1e-10:
            return pair, False
    # The common tangent checks all resolved dense minima. A smooth free
    # energy can still have an additional first-order transition for some
    # parameter choices; keep the lower envelope rather than forcing a root.
    gas, liquid, corner, _ = _coexistence(state, previous)
    if corner:
        raise RuntimeError("A smooth exclusion law cannot have a threshold-pinned subgradient state.")
    stable = np.array([gas, liquid])
    switched = pair is not None and abs(stable[1] - pair[1]) > 1e-5
    return stable, switched
