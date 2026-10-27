"""Supporting common tangents for a continuous free energy with a corner.

The compact branch is unchanged up to ``eos.phi_c``. The dense branch can
change its shape or exclusion layer beyond that imposed density threshold.
The EOS maps volume fraction to conserved density with ``eta_from_phi``.
"""

import numpy as np
from scipy.optimize import brentq

_CORNER_NODES, _CORNER_WEIGHTS = np.polynomial.legendre.leggauss(16)
_CORNER_T = (_CORNER_NODES + 1) / 2


class _DenseMinimum:
    """Find the lowest dense-branch grand potential, including its corner.

    Cached derivative samples bracket every resolved local minimum rather
    than assuming that this nonconvex free energy has a single minimum.
    """
    def __init__(self, reservoir):
        self.reservoir = reservoir
        self.corner = reservoir.eos.phi_c
        span = 1 - 1e-7 - self.corner
        phi_grid = self.corner + np.unique(np.concatenate([
            [0.0], np.geomspace(1e-10, span, 180), np.linspace(0, span, 180),
        ]))
        width = getattr(reservoir.eos, "width", None)
        if width is not None:
            transition = self.corner + width * np.geomspace(1e-3, 30, 100)
            phi_grid = np.unique(np.r_[phi_grid, transition[transition < 1 - 1e-7]])
        self.etas = reservoir.eos.eta_from_phi(phi_grid)
        values = reservoir.evaluate(self.etas)
        self.free, self.mu = values[:2]

    def __call__(self, mu, pressure):
        slope = self.mu - mu
        crossing = np.flatnonzero((slope[:-1] < 0) & (slope[1:] >= 0))
        candidates = [(self.free[0] - mu * self.corner + pressure, self.corner)]
        for i in crossing:
            eta = brentq(lambda x: self.reservoir.evaluate(x)[1] - mu,
                         self.etas[i], self.etas[i + 1], xtol=1e-14)
            value = self.reservoir.evaluate(eta)[0] - mu * eta + pressure
            candidates.append((value, eta))
        return min(candidates)


def _corner_pair(reservoir):
    gas = reservoir.gas
    corner = reservoir.eos.phi_c
    spin_g = brentq(lambda x: gas.slopes(x)[0], 1e-250, corner, xtol=1e-15)
    f_corner = gas.evaluate(corner)[0]

    def residual(log_eta):
        eta = np.exp(log_eta)
        gap = corner - eta
        if gap < 0.02 * corner:
            # Integral Taylor remainder avoids cancellation of f(c)-f(g)-mu*g
            # as the two concentrations approach the one-sided onset.
            trial = eta + _CORNER_T * gap
            curvature = np.array([gas.slopes(x)[0] / (x * (1 - x)**2) for x in trial])
            return np.dot(_CORNER_WEIGHTS / 2, (1 - _CORNER_T) * curvature)
        f, mu, _ = gas.evaluate(eta)
        return (f_corner - f - gap * mu) / gap**2

    eta = np.exp(brentq(residual, np.log(1e-250), np.log(spin_g), xtol=1e-12))
    return eta, corner


def _coexistence(reservoir, previous=None):
    gas = reservoir.gas
    gas_corner, corner = _corner_pair(reservoir)
    dense_min = _DenseMinimum(reservoir)
    _, mu, pressure = gas.evaluate(gas_corner)
    gap, dense = dense_min(mu, pressure)
    if gap >= -1e-10:
        # At a corner chemical potential is a subgradient, not the right-hand
        # derivative alone. Interpolate the two slopes and their decompositions.
        left = gas.evaluate(corner)[1]
        right = reservoir.evaluate(corner)[1]
        # For vanishing scaling the two slopes can round to the same number.
        # In that case try the interior solve instead of dividing by zero or
        # pretending that the supporting slope lies inside a resolved corner.
        fraction = (mu - left) / (right - left) if right > left else np.nan
        if -1e-7 <= fraction <= 1 + 1e-7:
            return gas_corner, corner, True, float(np.clip(fraction, 0, 1))
        # A minimum immediately beyond the corner can be too close for the
        # support tolerance. Continue with the interior solve in that case.

    def residual(log_gas):
        _, mu, pressure = gas.evaluate(np.exp(log_gas))
        return dense_min(mu, pressure)[0]

    upper = np.log(gas_corner)
    lower = np.log(min(gas_corner * 0.5, previous[0] * 0.8 if previous is not None else gas_corner * 0.5))
    log_floor = np.log(np.finfo(float).tiny)
    for _ in range(20):
        if residual(lower) > 0:
            break
        if lower <= log_floor:
            raise RuntimeError("The dilute concentration is below the representable density range.")
        lower = max(log_floor, upper - 2 * (upper - lower))
    else:
        raise RuntimeError("Could not bracket a supporting isotropic common tangent.")
    root = brentq(residual, lower, upper, xtol=1e-12)
    eta_g = np.exp(root)
    _, mu, pressure = gas.evaluate(eta_g)
    gap, eta_d = dense_min(mu, pressure)
    dense_values = reservoir.evaluate(eta_d)
    if (eta_d <= corner or abs(gap) > 1e-8
            or np.max(np.abs(dense_values[1:] - [mu, pressure])) > 1e-7):
        raise RuntimeError(f"Unresolved isotropic coexistence at phi_R={reservoir.phi_R:.10g}.")
    return eta_g, eta_d, False, 1.0
