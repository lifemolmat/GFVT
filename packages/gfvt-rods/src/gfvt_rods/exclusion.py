"""Sphere GFVT with a density-dependent PEG exclusion layer.

The protein volume, Carnahan--Starling reference and reservoir osmotic
equation are fixed. Only the depletion thickness changes:
``qx_eff = qx + s * max(phi - phi_c, 0)``. Chemical potential and pressure
include the derivative of that thickness with respect to protein density.

Free energies and pressures use kT/v_protein; chemical potentials use kT.
The density coordinate is the physical sphere volume fraction in both phases.
The threshold is a corner in the continuous free energy, not a smooth
critical point of the modified model.
"""

import numpy as np

from .rods import _real_scalar
from .spheres import SphereEOS, _coefficients


def _reference(eos, phi):
    """Carnahan--Starling [free energy, chemical potential, pressure]."""
    phi = np.asarray(phi)
    excess = (4 * phi - 3 * phi**2) / (1 - phi)**2
    free = phi * (eos.mu_constant + np.log(phi) - 1 + excess)
    mu, pressure = eos.reference(phi)
    return np.array([free, mu, pressure])


class ExclusionLayerEOS:
    def __init__(self, q, solvent, s, phi_c, mu_constant=0, *, nu=None):
        self.q = _real_scalar(q, "q", positive=True)
        self.s = _real_scalar(s, "s")
        self.phi_c = _real_scalar(phi_c, "phi_c", positive=True)
        if self.phi_c >= 1:
            raise ValueError("phi_c must be less than 1.")
        self.solvent = solvent
        self.mu_constant = mu_constant
        self.base = SphereEOS(self.q, solvent, mu_constant, nu=nu)

    @staticmethod
    def eta_from_phi(phi):
        return np.asarray(phi)

    @staticmethod
    def phi_from_eta(eta):
        return np.asarray(eta)

    def at(self, phi_R):
        return _ExclusionReservoir(self, phi_R)

    def layer_increment(self, phi):
        """Sharp layer increment and its right derivative at the threshold."""
        phi = np.asarray(phi)
        return self.s * np.maximum(phi - self.phi_c, 0), self.s * (phi >= self.phi_c)


class _SphereFreeEnergyReservoir:
    """Add the semigrand free energy to the unchanged sphere reservoir."""

    def __init__(self, reservoir):
        self.reservoir = reservoir
        self.eos = reservoir.eos
        self.phi_R = reservoir.phi_R

    def reference(self, phi):
        return _reference(self.eos, phi)

    def evaluate(self, phi):
        phi = np.asarray(phi)
        y, beta, Q1 = self.reservoir._beta(phi[..., None], self.reservoir.coefficients)
        alpha = (1 - phi[..., None]) * beta
        g = beta * (1 + (1 + y) * Q1)
        h = beta * (1 + y * Q1)
        weights = self.reservoir.weighted_dpi
        return self.reference(phi) + np.array([
            -np.sum(weights * alpha, axis=-1),
            np.sum(weights * g, axis=-1),
            np.sum(weights * h, axis=-1),
        ])

    def partition(self, phi):
        return self.reservoir.partition(np.asarray(phi))

    def slopes(self, phi):
        return self.reservoir.slopes(phi)


class _ExclusionReservoir:
    def __init__(self, eos, phi_R):
        self.eos = eos
        self.phi_R = float(phi_R)
        self.sphere = eos.base.at(phi_R)
        self.gas = _SphereFreeEnergyReservoir(self.sphere)
        self.qx, self.dpi = self.sphere.endpoint_qx, self.sphere.endpoint_dpi
        self.grid_qx = self.sphere.qx
        # Pressure normalization stays fixed as the exclusion layer changes.
        self.weights = self.sphere.weighted_dpi

    def reference(self, phi):
        return _reference(self.eos.base, phi)

    def qx_at(self, phi):
        return self.qx + self.eos.layer_increment(phi)[0]

    def _partition(self, phi, qx):
        phi = np.asarray(phi)[..., None]
        offset, slope = self.eos.layer_increment(phi)
        qx = np.asarray(qx) + offset
        a, b, c = _coefficients(qx)
        aq = 3 + 6 * qx + 3 * qx**2
        bq = 9 * qx + 9 * qx**2
        cq = 9 * qx**2
        y = phi / (1 - phi)
        Q = a * y + b * y**2 + c * y**3
        Q_prime = ((a + 2 * b * y + 3 * c * y**2) / (1 - phi)**2
                   + slope * (aq * y + bq * y**2 + cq * y**3))
        beta = np.exp(-Q)
        alpha = (1 - phi) * beta
        g = beta * (1 + (1 - phi) * Q_prime)
        return np.array([alpha, g, alpha + phi * g])

    def evaluate(self, phi):
        alpha, g, h = self._partition(phi, self.grid_qx)
        return self.reference(phi) + np.array([
            -np.sum(self.weights * alpha, axis=-1),
            np.sum(self.weights * g, axis=-1),
            np.sum(self.weights * h, axis=-1),
        ])

    def partition(self, phi):
        return self._partition(phi, self.qx)[..., 0]
