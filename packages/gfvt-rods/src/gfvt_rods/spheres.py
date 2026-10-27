"""Cached sphere GFVT integrals with the original fluid/crystal references.

The Carnahan--Starling fluid and crystal equations, depletion coefficients,
and osmotic-pressure normalization match PD.GFVT. Only the quadrature changes.
An optional phase q evaluates these same equations at a different fixed size
ratio. Density-dependent exclusion layers are implemented in exclusion.py.
"""

import numpy as np
from scipy.optimize import least_squares

from ._quadrature import _T, _W


def _coefficients(qx):
    return 3 * qx + 3 * qx**2 + qx**3, 4.5 * qx**2 + 3 * qx**3, 3 * qx**3


def critical_endpoint(solvent, *, nu, kuhn_crossover=None, protein_radius=None):
    """Locate the fluid critical point that also coexists with a crystal.

    Solve both critical conditions and fluid/crystal equality of mu and Pi,
    as in Fleer & Tuinier (2008), Sections 2.4 and 6.3. The published q_R*
    values are model results, not constants to reuse after changing nu.
    Densities and q use this package's physical sphere and dilute PEG radii.
    """
    def residual(values):
        phi_fluid, phi_solid, phi_R, q = values
        state = SphereEOS(q, solvent, nu=nu, kuhn_crossover=kuhn_crossover,
                          protein_radius=protein_radius).at(phi_R)
        return np.r_[state.slopes(phi_fluid),
                     state.evaluate(phi_fluid) - state.evaluate(phi_solid, solid=True)]

    sol = least_squares(
        residual, [0.317, 0.594, 0.428, 0.388],
        bounds=([1e-6, 0.5, 1e-6, 0.01], [0.7, 0.741-1e-8, 500, 10]),
        xtol=1e-13, ftol=1e-13, gtol=1e-13,
    )
    if (not sol.success or np.max(np.abs(sol.fun)) > 1e-8
            or np.any(sol.active_mask) or sol.x[0] >= sol.x[1]):
        raise RuntimeError(f"Could not resolve the critical endpoint for {solvent=}, {nu=}.")
    return dict(zip(("phi_fluid", "phi_solid", "phi_R", "q"), map(float, sol.x)))


class SphereEOS:
    def __init__(self, q, solvent, mu_constant=0, *, nu=None,
                 kuhn_crossover=None, protein_radius=None):
        from .polymer import PolymerScaling
        self.q = q
        self.solvent = solvent
        self.mu_constant = mu_constant
        self.polymer = PolymerScaling(solvent, nu, kuhn_crossover, protein_radius)

    def reference(self, phi, solid=False):
        if solid:
            gap = 1 - phi / 0.741
            mu = self.mu_constant + 2.1178 + 3 * np.log(phi / gap) + 3 / gap
            pressure = 3 * phi / gap
        else:
            mu = self.mu_constant + np.log(phi) + (3 - phi) / (1 - phi)**3 - 3
            pressure = (phi + phi**2 + phi**3 - phi**4) / (1 - phi)**3
        return np.array([mu, pressure])

    def at(self, phi_R):
        return _SphereReservoir(self, phi_R)


class _SphereReservoir:
    def __init__(self, eos, phi_R):
        self.eos = eos
        self.phi_R = float(phi_R)
        self.grid = self.phi_R * _T**4
        self.weights = _W * (4 * self.phi_R * _T**3)
        self.qx, self.dpi = eos.polymer.qxf(eos.q, self.grid)
        self.coefficients = _coefficients(self.qx)
        self.weighted_dpi = self.weights * self.dpi
        self.endpoint_qx, self.endpoint_dpi = eos.polymer.qxf(eos.q, self.phi_R)
        self.endpoint_coefficients = _coefficients(self.endpoint_qx)

    def _phase_coefficients(self, q):
        if q is None or q == self.eos.q:
            return self.coefficients, self.weighted_dpi
        # For scaled droplets reuse the reservoir-only powers as q changes.
        cross = self.eos.polymer.kuhn_crossover
        if self.eos.solvent in ("good", "theta") and (cross is None or cross.s == 1):
            ratio = q / self.eos.q
            exponent = self.eos.polymer.curvature_power
            qx = self.qx * ratio**exponent
            dpi = self.dpi / ratio**3
        else:
            qx, dpi = self.eos.polymer.qxf(q, self.grid)
        return _coefficients(qx), self.weights * dpi

    @staticmethod
    def _beta(phi, coefficients):
        y = phi / (1 - phi)
        a, b, c = coefficients
        beta = np.exp(-(a * y + b * y**2 + c * y**3))
        Q1 = a + 2 * b * y + 3 * c * y**2
        return y, beta, Q1

    def evaluate(self, phi, q=None, solid=False):
        coefficients, weights = self._phase_coefficients(q)
        y, beta, Q1 = self._beta(phi, coefficients)
        g = beta * (1 + (1 + y) * Q1)
        h = beta * (1 + y * Q1)
        return self.eos.reference(phi, solid=solid) + np.array([
            np.dot(weights, g), np.dot(weights, h),
        ])

    def semigrand(self, phi, *, solid=False):
        """Omega*v_BSA/(V*kBT), including the polymer reservoir background."""
        if phi == 0:
            return -np.sum(self.weighted_dpi)
        mu0, p0 = self.eos.reference(phi, solid=solid)
        _, beta, _ = self._beta(phi, self.coefficients)
        return phi * mu0 - p0 - np.dot(self.weighted_dpi, (1-phi)*beta)

    def slopes(self, phi):
        """First two pressure derivatives with respect to y = phi/(1-phi)."""
        y, beta, Q1 = self._beta(phi, self.coefficients)
        _, b, c = self.coefficients
        Q2 = 2 * b + 6 * c * y
        beta2 = beta * (Q1**2 - Q2)
        beta3 = beta * (-Q1**3 + 3 * Q1 * Q2 - 6 * c)
        return np.array([
            (1 + y)**-2 + 8 * y + 6 * y**2 - y * np.dot(self.weighted_dpi, beta2),
            -2 * (1 + y)**-3 + 8 + 12 * y - np.dot(self.weighted_dpi, beta2 + y * beta3),
        ])

    def partition(self, phi, q=None):
        if q is None or q == self.eos.q:
            coefficients = self.endpoint_coefficients
        else:
            coefficients = _coefficients(self.eos.polymer.qxf(q, self.phi_R)[0])
        y, beta, Q1 = self._beta(phi, coefficients)
        return np.array([
            (1 - phi) * beta,
            beta * (1 + (1 + y) * Q1),
            beta * (1 + y * Q1),
        ])
