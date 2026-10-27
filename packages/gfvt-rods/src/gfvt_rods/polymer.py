"""Shared polymer scaling for the GFVT reservoir and its diagnostics.

Lekkerkerker, Tuinier & Vis (2024), Chapter 4, Eqs. 4.8--4.35:
https://doi.org/10.1007/978-3-031-52131-7_4

An explicit nu derives every concentration exponent from gamma=nu/(3nu-1).
nu=None preserves the published rounded numerical closure used by old runs.
The solvent-specific crossover amplitudes and spherical curvature fits are
independent inputs; varying nu alone is not a solvent-quality crossover theory.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

GOOD_SOLVENT_NU = 0.588


@dataclass(frozen=True)
class KuhnCrossover:
    """Heuristic pre-curvature crossover; b is in metres, not a hard floor.

    s retains this fraction of the high-concentration depletion exponent.
    s=1 is an exact baseline branch. Reservoir pressure is never modified.
    """

    b: float = 0.75e-9
    m: float = 4.0
    s: float = 1.0

    def __post_init__(self):
        for name in ("b", "m", "s"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"Kuhn crossover {name} must be positive and finite.")
        if self.s > 1:
            raise ValueError("Kuhn crossover s must satisfy 0 < s <= 1.")


@dataclass(frozen=True)
class PolymerScaling:
    solvent: str
    nu: float | None = None
    kuhn_crossover: KuhnCrossover | None = None
    protein_radius: float | None = None

    def __post_init__(self):
        if self.kuhn_crossover is not None:
            if not isinstance(self.kuhn_crossover, KuhnCrossover):
                raise TypeError("kuhn_crossover must be a KuhnCrossover instance or None.")
            if self.solvent != "good":
                raise ValueError("Kuhn crossover currently requires solvent='good'.")
            radius = self.protein_radius
            if (isinstance(radius, (bool, np.bool_)) or not isinstance(radius, Real)
                    or not np.isfinite(radius) or radius <= 0):
                raise ValueError("Kuhn crossover requires positive finite protein_radius in metres.")
        if self.nu is not None:
            if (isinstance(self.nu, (bool, np.bool_)) or not isinstance(self.nu, Real)
                    or not np.isfinite(self.nu) or not 0.5 <= self.nu < 1):
                raise ValueError("nu must be a finite real scalar with 0.5 <= nu < 1.")
            if self.solvent not in ("good", "RG", "theta"):
                raise ValueError("nu requires a nonideal polymer solvent model.")
            if self.solvent == "theta" and self.nu != 0.5:
                raise ValueError("The theta-solvent closure requires nu=0.5.")

    @property
    def gamma(self):
        if self.nu is not None:
            return self.nu / (3 * self.nu - 1)
        return 0.77 if self.solvent in ("good", "RG") else 1.0

    @property
    def pressure_power(self):
        return 3 * self.gamma

    @property
    def zeta(self):
        if self.solvent in ("good", "RG"):
            # The legacy derivative is 3.73*r**1.31. Integrate that exact
            # closure for legacy pressure plots, instead of mixing with 1.62.
            return 1.62 if self.nu is not None else 3.73 / self.pressure_power
        return 4.10 if self.solvent == "theta" else 0.0

    @property
    def depletion_power(self):
        return 2 * self.gamma

    @property
    def beta(self):
        return 6.02 if self.solvent == "theta" else 3.95

    @property
    def curvature_power(self):
        return 0.9 if self.solvent == "theta" else 0.88

    def pressure(self, phi_R):
        """Polymer osmotic pressure in kT/v_PEG; phi_R uses fixed dilute Rg."""
        phi_R = np.asarray(phi_R, dtype=float)
        return phi_R + self.zeta * phi_R**self.pressure_power

    def pressure_derivative(self, phi_R):
        phi_R = np.asarray(phi_R, dtype=float)
        if self.nu is None and self.solvent in ("good", "RG"):
            return 1 + 3.73 * phi_R**1.31
        if self.nu is None and self.solvent == "theta":
            return 1 + 12.3 * phi_R**2
        return 1 + self.pressure_power * self.zeta * phi_R**(self.pressure_power - 1)

    def depletion(self, q, phi_R, *, exponent=None, semidilute=False):
        """Spherical depletion thickness / protein radius (curvature fitted)."""
        phi_R = np.asarray(phi_R, dtype=float)
        crossover = self.kuhn_crossover
        if crossover is not None:
            if semidilute:
                raise ValueError("Kuhn crossover is not defined for the semidilute-only closure.")
            if not np.isfinite(q) or q <= 0 or np.any(~np.isfinite(phi_R)) or np.any(phi_R < 0):
                raise ValueError("Kuhn crossover requires q > 0 and finite phi_R >= 0.")
            if crossover.s != 1:
                log_ratio = self._log_length_ratio(q, phi_R, exponent=exponent)
                return 0.865 * np.exp(self.curvature_power * (np.log(q) + log_ratio))
        if self.solvent not in ("good", "RG", "theta"):
            return q * np.ones_like(phi_R)
        exponent = self.depletion_power if exponent is None else exponent
        denominator = (0 if semidilute else 1) + self.beta * phi_R**exponent
        with np.errstate(divide="ignore", invalid="ignore"):
            dq = q / np.sqrt(denominator)
            if self.solvent == "RG":
                return (1 + 3 * dq + 2.73 * dq**2 - 0.0975 * dq**3)**(1 / 3) - 1
            return (0.938 if self.solvent == "theta" else 0.865) * dq**self.curvature_power

    def _log_length_ratio(self, q, phi_R, *, exponent=None):
        """log(ell/Rg), evaluated without powers of extreme length ratios."""
        exponent = self.depletion_power if exponent is None else exponent
        with np.errstate(divide="ignore"):
            log_denom = np.logaddexp(0.0, np.log(self.beta) + exponent * np.log(phi_R))
        log_ratio = -0.5 * log_denom
        cross = self.kuhn_crossover
        if cross is not None and cross.s != 1:
            log_b_rg = np.log(cross.b) - np.log(self.protein_radius) - np.log(q)
            # logaddexp divided by m avoids explicitly forming (b/ell)**m.
            correction = (np.logaddexp(0.0, cross.m * (log_b_rg-log_ratio))
                          - np.logaddexp(0.0, cross.m * log_b_rg)) / cross.m
            log_ratio = log_ratio + (1-cross.s) * correction
        return log_ratio

    def underlying_length(self, q, phi_R, *, exponent=None):
        """Pre-curvature ell in metres, for the full good-solvent closure.

        This diagnostic is not asserted to be a measured correlation length.
        An explicit exponent preserves a selected electrostatic model closure.
        """
        if self.solvent != "good" or self.protein_radius is None:
            raise ValueError("underlying_length requires good solvent and protein_radius.")
        phi_R = np.asarray(phi_R, dtype=float)
        if (not np.isfinite(q) or q <= 0 or not np.isfinite(self.protein_radius)
                or self.protein_radius <= 0 or np.any(~np.isfinite(phi_R)) or np.any(phi_R < 0)):
            raise ValueError("Require positive radii and finite phi_R >= 0.")
        return q * self.protein_radius * np.exp(self._log_length_ratio(q, phi_R, exponent=exponent))

    def qxf(self, q, phi_R):
        # Keep the historical ideal-sphere normalization; nonideal closures
        # use v_protein/v_PEG=q^-3 as in the book.
        divisor = q**3 if self.solvent in ("good", "theta", "RG") else 1.0
        return self.depletion(q, phi_R), self.pressure_derivative(phi_R) / divisor
