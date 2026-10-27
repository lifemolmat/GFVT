"""Isotropic, fixed-shape spherocylinder GFVT.

Reference and insertion free energies: Lekkerkerker, Tuinier & Vis (2024),
https://doi.org/10.1007/978-3-031-52131-7_8, Eqs. 8.38--8.54, rho[f]=1.
For nonideal polymers the existing sphere-derived depletion thickness at
radius D/2 is an explicit approximation, not a rod-specific polymer theory.
The reservoir osmotic pressure uses the actual spherocylinder volume.
"""

import numpy as np
from scipy.optimize import brentq, least_squares
from tqdm import tqdm

from ._quadrature import _T, _W
from .polymer import PolymerScaling


def _real_scalar(value, name, *, positive=False):
    if (not np.isscalar(value) or not np.isreal(value)
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} real scalar.")
    return float(value)


class IsotropicRodEOS:
    """SPT free energies in units kT/v_rod; chemical potential in kT.

    D is the diameter and L excludes the two hemispherical end caps, in metres.
    The additive chemical-potential constant has no effect on coexistence.
    """

    def __init__(self, D, L, Rpol, solvent, mu_constant=0, *, nu=None):
        self.polymer = PolymerScaling(solvent, nu)
        self.D = _real_scalar(D, "D", positive=True)
        self.L = _real_scalar(L, "L")
        self.Rpol = _real_scalar(Rpol, "Rpol", positive=True)
        self.solvent = solvent
        self.mu_constant = mu_constant
        self.gamma = 1 + self.L / self.D
        self.volume = np.pi * self.D**2 * self.L / 4 + np.pi * self.D**3 / 6
        self.polymer_volume = 4 * np.pi * self.Rpol**3 / 3
        self.q = 2 * self.Rpol / self.D
        self.A = 3 + 3 * (self.gamma - 1)**2 / (3 * self.gamma - 1)
        self.B = 12 * self.gamma**3 / (3 * self.gamma - 1)**2

    def depletion(self, phi_R):
        # Only the thickness is borrowed. q^-3 alone would use the volume of
        # the end-cap sphere rather than the volume of the complete rod.
        qx = self.polymer.depletion(self.q, phi_R)
        osmotic = self.polymer.pressure_derivative(phi_R)
        return qx, osmotic * self.volume / self.polymer_volume

    def coefficients(self, qx):
        gamma = self.gamma
        denominator = 3 * gamma - 1
        a = (6 * gamma * qx + 3 * (gamma + 1) * qx**2 + 2 * qx**3) / denominator
        b = 0.5 * (6 * gamma / denominator)**2 * qx**2 + 2 * self.A / denominator * qx**3
        c = 2 * self.B / denominator * qx**3
        return a, b, c

    def reference(self, phi):
        y = phi / (1 - phi)
        A, B = self.A, self.B
        log_y = np.log(phi) - np.log1p(-phi)
        f = phi * (self.mu_constant - 1 + log_y + A * y + 0.5 * B * y**2)
        mu = self.mu_constant + log_y + (1 + 2 * A) * y + (A + 1.5 * B) * y**2 + B * y**3
        p = y + A * y**2 + B * y**3
        return np.array([f, mu, p])

    def at(self, phi_R):
        return _RodReservoir(self, phi_R)


class _RodReservoir:
    def __init__(self, eos, phi_R):
        self.eos = eos
        self.phi_R = float(phi_R)
        grid = self.phi_R * _T**4
        qx, dpi = eos.depletion(grid)
        self.coefficients = eos.coefficients(qx)
        self.weights = _W * (4 * self.phi_R * _T**3) * dpi
        self.qx, self.dpi = eos.depletion(self.phi_R)
        self.endpoint_coefficients = eos.coefficients(self.qx)

    @staticmethod
    def _beta(phi, coefficients):
        y = phi / (1 - phi)
        a, b, c = coefficients
        beta = np.exp(-(a * y + b * y**2 + c * y**3))
        Q1 = a + 2 * b * y + 3 * c * y**2
        return y, beta, Q1

    def evaluate(self, phi):
        y, beta, Q1 = self._beta(phi, self.coefficients)
        alpha = (1 - phi) * beta
        g = beta * (1 + (1 + y) * Q1)
        h = beta * (1 + y * Q1)
        return self.eos.reference(phi) + np.array([
            -np.dot(self.weights, alpha),
            np.dot(self.weights, g),
            np.dot(self.weights, h),
        ])

    def slopes(self, phi):
        y, beta, Q1 = self._beta(phi, self.coefficients)
        _, b, c = self.coefficients
        Q2 = 2 * b + 6 * c * y
        beta2 = beta * (Q1**2 - Q2)
        beta3 = beta * (-Q1**3 + 3 * Q1 * Q2 - 6 * c)
        A, B = self.eos.A, self.eos.B
        return np.array([
            1 + 2 * A * y + 3 * B * y**2 - y * np.dot(self.weights, beta2),
            2 * A + 6 * B * y - np.dot(self.weights, beta2 + y * beta3),
        ])

    def partition(self, phi):
        y, beta, Q1 = self._beta(phi, self.endpoint_coefficients)
        return np.array([(1 - phi) * beta, beta * (1 + (1 + y) * Q1), beta * (1 + y * Q1)])


def _critical_point(eos, guess):
    def residual(log_values):
        phi, phi_R = np.exp(log_values)
        y = phi / (1 - phi)
        normalization = [1 + 2 * eos.A * y + 3 * eos.B * y**2, 2 * eos.A + 6 * eos.B * y]
        return eos.at(phi_R).slopes(phi) / normalization

    lower, upper = np.log([1e-7, 1e-7]), np.log([0.95, 500.0])
    # The notebook guesses were selected for spheres. Extra seeds cover the
    # shifted rod critical point without anchoring it to the old sphere CP.
    seeds = [guess, (0.1, 3), (0.05, 10), (0.2, 1), (0.02, 30)]
    for seed in seeds:
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (2,) or np.any(~np.isfinite(seed)) or np.any(seed <= 0):
            continue
        sol = least_squares(residual, np.clip(np.log(seed), lower + 1e-8, upper - 1e-8),
                            bounds=(lower, upper), gtol=1e-12, ftol=1e-12,
                            xtol=1e-12, max_nfev=300)
        if sol.success and not np.any(sol.active_mask) and np.max(np.abs(sol.fun)) < 1e-9:
            phi_c, phi_R_c = np.exp(sol.x)
            # The required endpoint has a stable fluid just below it and an
            # instability just above it, not an unrelated stationary root.
            if (eos.at(phi_R_c * (1 - 1e-4)).slopes(phi_c)[0] > 0
                    and eos.at(phi_R_c * (1 + 1e-4)).slopes(phi_c)[0] < 0):
                return float(phi_c), float(phi_R_c)
    raise RuntimeError("Could not locate an isotropic rod critical point (searched 0 < phi_R < 500).")


def _solve_pair(reservoir, phi_c, previous):
    floor, ceiling = 1e-250, 1 - 1e-7
    slope = lambda phi: reservoir.slopes(phi)[0]
    center = phi_c
    if slope(center) >= 0:
        candidates = np.linspace(1e-5, 0.95, 200)
        center = min(candidates, key=slope)
    if slope(center) >= 0:
        raise RuntimeError("Could not resolve the isotropic spinodal interval.")
    spin_g = brentq(slope, floor, center, xtol=1e-15)
    spin_l = brentq(slope, center, ceiling, xtol=1e-15)
    midpoint = (spin_g + spin_l) / 2
    half_gap = np.sqrt(3) * (spin_l - spin_g) / 2
    seed = [max(midpoint - half_gap, spin_g / 2), min(midpoint + half_gap, (1 + spin_l) / 2)]
    lower, upper = np.log([floor, spin_l]), np.log([spin_g, ceiling])

    def residual(log_phi):
        gas, liquid = np.exp(log_phi)
        return (reservoir.evaluate(gas)[1:] - reservoir.evaluate(liquid)[1:]) / (liquid - gas)

    for guess in (previous, seed):
        if guess is None:
            continue
        sol = least_squares(residual, np.clip(np.log(guess), lower + 1e-10, upper - 1e-10),
                            bounds=(lower, upper), x_scale="jac", gtol=1e-12,
                            ftol=1e-12, xtol=1e-12, max_nfev=150)
        if sol.success and not np.any(sol.active_mask) and np.max(np.abs(sol.fun)) < 1e-8:
            return np.exp(sol.x)
    raise RuntimeError(f"Could not find distinct isotropic fluids at phi_R={reservoir.phi_R:.10g}.")


def calculate_rods(pd, PhiP_max, *, D=None, L=0, progress=None, verbose=None):
    """Compute into local arrays and update PD only after a successful solve."""
    D = 2 * pd.Rpr_in if D is None else D
    eos = IsotropicRodEOS(D, L, pd.Rpol_in, pd.solvent, nu=getattr(pd, "nu", None))
    PhiP_max = _real_scalar(PhiP_max, "PhiP_max", positive=True)
    if pd.N_steps < 2:
        raise ValueError("N_steps must be at least 2 to include the critical point and upper limit.")
    progress = pd.progress if progress is None else progress
    verbose = pd.verbose if verbose is None else verbose
    vbsa = eos.volume * 6.022e23
    vpeg = eos.polymer_volume * 6.022e23
    phix_bsa, phix_peg = 1 / vbsa, 1 / vpeg
    T = 273.15 + pd.TC
    Lam = 6.626e-34 / np.sqrt(2 * np.pi * pd.mbsa * 1000 * 1.38e-23 * T / 6.022e23)
    eos.mu_constant = np.log(Lam**3 / vbsa)
    phi_c, phi_R_c = _critical_point(eos, pd.guess_c)
    if PhiP_max < phi_R_c:
        raise ValueError(f"PhiP_max={PhiP_max:g} must be >= the isotropic rod critical reservoir concentration {phi_R_c:.10g}.")
    if verbose:
        print(f"Isotropic rods: D={eos.D * 1e9:g} nm, L={eos.L * 1e9:g} nm; critical phi={phi_c:.8g}, phi_R={phi_R_c:.8g}")

    result = {name: np.empty(pd.N_steps) for name in pd.RETURN_FIELDS
              if name not in ("cp", "tp", "cp_list", "tp_list", "params")}
    current, pair = phi_R_c, None
    seed_step = max(phi_R_c, 1e-3) * 0.01
    min_step = max(phi_R_c, 1.0) * 1e-10
    pressure_units = phix_bsa * 8.314 * T  # Pa: mM equals mol/m^3
    for k, target in enumerate(tqdm(np.linspace(phi_R_c, PhiP_max, pd.N_steps),
                                    desc="Isotropic rod binodal", disable=not progress)):
        attempts = 0
        while current < target:
            step = min(target - current, max(seed_step, (current - phi_R_c) / 2))
            while True:
                attempts += 1
                if attempts > 100:
                    raise RuntimeError(f"Rod continuation stalled near phi_R={current:.10g}.")
                next_phi_R = min(target, current + step)
                try:
                    next_pair = _solve_pair(eos.at(next_phi_R), phi_c, pair)
                except RuntimeError:
                    step /= 2
                    if step < min_step or current + step == current:
                        raise
                else:
                    current, pair = next_phi_R, next_pair
                    break
        gas, liquid = (phi_c, phi_c) if pair is None else pair
        j = pd.N_steps - 1 - k
        result["phi_R"][j] = target
        reservoir = eos.at(target)
        for phase, phi in (("Sup", gas), ("Drop", liquid)):
            _, mu, p = reservoir.evaluate(phi)
            _, mu0, p0 = eos.reference(phi)
            alpha, g, h = reservoir.partition(phi)
            for key, value in {
                f"Pr_{phase}_p": phi * phix_bsa,
                f"Pol_{phase}_p": alpha * target * phix_peg,
                f"a_{phase}_p": alpha,
                f"qx_{phase}_p": reservoir.qx,
                f"dPidPhiP_{phase}_p": reservoir.dpi,
                f"Pi_{phase}": p * pressure_units,
                f"P0_{phase}": p0 * pressure_units,
                f"Pp_{phase}": (p - p0) * pressure_units,
                f"Mu_{phase}": mu,
                f"Mu0_{phase}": mu0,
                f"MuR_{phase}": mu - mu0,
                f"g_{phase}": g,
                f"h_{phase}": h,
            }.items():
                result[key][j] = value

    pd._clear_scaled_rod_metadata()
    pd.geometry = "isotropic_rods"
    pd.D, pd.L, pd.aspect_ratio = eos.D, eos.L, eos.gamma
    pd.depletion_model = "diameter-based spherical thickness approximation" if pd.solvent in ("good", "theta", "RG") else "ideal penetrable spheres"
    pd.q, pd.vbsa, pd.vpeg = eos.q, vbsa, vpeg
    pd.phix_bsa, pd.phix_peg = phix_bsa, phix_peg
    pd.PhiP_min, pd.PhiP_max = phi_R_c, PhiP_max
    pd.outguess_c, pd.outguess_tp = [phi_c, phi_R_c], None
    pd.s = 0
    pd.q_Drop_p = np.full(pd.N_steps, eos.q)
    pd.q_Sup_p = np.full(pd.N_steps, eos.q)
    result.update(cp=[phi_c * phix_bsa, eos.at(phi_R_c).partition(phi_c)[0] * phi_R_c * phix_peg],
                  tp=None, cp_list=[], tp_list=[], params=pd._build_params())
    pd._assign_return_values([result[name] for name in pd.RETURN_FIELDS])
    return pd
