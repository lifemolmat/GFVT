"""Isotropic coexistence for an imposed, density-dependent rod length.

The compact branch has L=0; the dense branch has L=D*s*(phi-phi_c).
phi_c is the fixed-shape sphere-SPT critical volume fraction. There is no
additional conformational free energy. This is a constrained phenomenological
model, not a molecular model of BSA unfolding. Its free energy is continuous
but has a corner at phi_c, which must be included in the common-tangent solve.

Free energies and pressures use kT/v_sphere, eta=n*v_sphere is the conserved
density coordinate, and chemical potentials include all shape derivatives.
"""

import numpy as np
from tqdm import tqdm

from .rods import IsotropicRodEOS, _critical_point, _real_scalar, _T, _W, calculate_rods
# Re-export private solver helpers for existing diagnostic imports.
from .common_tangent import _DenseMinimum, _corner_pair, _coexistence


class ScaledIsotropicRodEOS:
    def __init__(self, D, Rpol, solvent, s, phi_c, mu_constant=0, *, nu=None):
        self.base = IsotropicRodEOS(D, 0, Rpol, solvent, mu_constant, nu=nu)
        self.D, self.Rpol, self.solvent = self.base.D, Rpol, solvent
        self.s = _real_scalar(s, "s")
        self.phi_c = _real_scalar(phi_c, "phi_c", positive=True)
        self.k = 1.5 * self.s
        self.mapping_factor = 1 - self.k * self.phi_c
        if self.phi_c >= 1 or self.mapping_factor <= 0:
            raise ValueError("s must satisfy 1 - 1.5*s*phi_c > 0 for a monotonic concentration-to-volume-fraction mapping.")
        self.volume = self.base.volume
        self.mu_constant = mu_constant

    def volume_factor(self, phi):
        return 1 + self.k * (np.asarray(phi) - self.phi_c)

    def eta_from_phi(self, phi):
        return np.asarray(phi) / self.volume_factor(phi)

    def phi_from_eta(self, eta):
        eta = np.asarray(eta)
        return eta * self.mapping_factor / (1 - self.k * eta)

    def length(self, phi):
        return self.D * self.s * (np.asarray(phi) - self.phi_c)

    def at(self, phi_R):
        return _ScaledRodReservoir(self, phi_R)


class _ScaledRodReservoir:
    def __init__(self, eos, phi_R):
        self.eos = eos
        self.phi_R = float(phi_R)
        self.gas = eos.base.at(phi_R)
        self.qx, self.dpi = eos.base.depletion(self.phi_R)
        self.grid_qx, dpi = eos.base.depletion(self.phi_R * _T**4)
        # The reference particle volume is fixed, even when the dense particle
        # changes shape. Reservoir pressure is identical in the two phases.
        self.weights = _W * (4 * self.phi_R * _T**3) * dpi

    def _geometry(self, eta):
        eta = np.asarray(eta)
        phi = self.eos.phi_from_eta(eta)
        phi_prime = self.eos.mapping_factor / (1 - self.eos.k * eta)**2
        gamma = 1 + self.eos.s * (phi - self.eos.phi_c)
        gamma_prime = self.eos.s * phi_prime
        d = 3 * gamma - 1
        A = 3 + 3 * (gamma - 1)**2 / d
        B = 12 * gamma**3 / d**2
        Ag = 3 * (gamma - 1) * (3 * gamma + 1) / d**2
        Bg = 36 * gamma**2 * (gamma - 1) / d**3
        return phi, phi_prime, gamma, gamma_prime, A, B, Ag, Bg

    def reference(self, eta):
        eta = np.asarray(eta)
        phi, phi_prime, _, gamma_prime, A, B, Ag, Bg = self._geometry(eta)
        y = phi / (1 - phi)
        y_prime = phi_prime / (1 - phi)**2
        excess = -np.log1p(-phi) + A * y + 0.5 * B * y**2
        excess_prime = (phi_prime / (1 - phi) + (A + B * y) * y_prime
                        + (Ag * y + 0.5 * Bg * y**2) * gamma_prime)
        omega = eta * (self.eos.mu_constant + np.log(eta) - 1 + excess)
        mu = self.eos.mu_constant + np.log(eta) + excess + eta * excess_prime
        p = eta + eta**2 * excess_prime
        return np.array([omega, mu, p])

    def _partition(self, eta, qx):
        eta = np.asarray(eta)
        phi, phi_prime, gamma, gamma_prime, A, B, Ag, Bg = self._geometry(eta)
        # A trailing dimension handles polymer-reservoir quadrature for either
        # one density or a vector of trial densities.
        phi, phi_prime, gamma, gamma_prime, A, B, Ag, Bg = (
            np.asarray(x)[..., None] for x in (phi, phi_prime, gamma, gamma_prime, A, B, Ag, Bg)
        )
        d = 3 * gamma - 1
        qx = np.asarray(qx)
        numerator = 6 * gamma * qx + 3 * (gamma + 1) * qx**2 + 2 * qx**3
        a = numerator / d
        ag = ((6 * qx + 3 * qx**2) * d - 3 * numerator) / d**2
        u = 6 * gamma / d
        b = 0.5 * u**2 * qx**2 + 2 * A / d * qx**3
        bg = -6 * u / d**2 * qx**2 + 2 * (Ag / d - 3 * A / d**2) * qx**3
        c = 2 * B / d * qx**3
        cg = 2 * (Bg / d - 3 * B / d**2) * qx**3
        y = phi / (1 - phi)
        y_prime = phi_prime / (1 - phi)**2
        Q = a * y + b * y**2 + c * y**3
        Q_prime = ((a + 2 * b * y + 3 * c * y**2) * y_prime
                   + (ag * y + bg * y**2 + cg * y**3) * gamma_prime)
        beta = np.exp(-Q)
        alpha = (1 - phi) * beta
        derivative = beta * (-phi_prime - (1 - phi) * Q_prime)
        return np.array([alpha, -derivative, alpha - eta[..., None] * derivative])

    def evaluate(self, eta):
        alpha, g, h = self._partition(eta, self.grid_qx)
        return self.reference(eta) + np.array([
            -np.sum(alpha * self.weights, axis=-1),
            np.sum(g * self.weights, axis=-1),
            np.sum(h * self.weights, axis=-1),
        ])

    def partition(self, eta):
        return self._partition(eta, self.qx)[..., 0]


def _install_shape_metadata(pd, s, phi_c, reference_critical, phi_drop, phi_sup, boundary, fraction):
    pd.geometry = "isotropic_rods_scaled"
    pd.s = s
    pd.reference_cp_phi = phi_c
    pd.reference_outguess_c = list(reference_critical)
    pd.Phi_Drop_p = np.asarray(phi_drop)
    pd.Phi_Sup_p = np.asarray(phi_sup)
    pd.L_Drop_p = pd.D * s * (pd.Phi_Drop_p - phi_c)
    pd.L_Sup_p = np.zeros(pd.N_steps)
    pd.vbsa_Drop_p = pd.vbsa * (1 + 1.5 * s * (pd.Phi_Drop_p - phi_c))
    pd.vbsa_Sup_p = np.full(pd.N_steps, pd.vbsa)
    pd.phix_bsa_Drop_p = 1 / pd.vbsa_Drop_p
    pd.phix_bsa_Sup_p = 1 / pd.vbsa_Sup_p
    pd.boundary_Drop_p = np.asarray(boundary, dtype=bool)
    pd.corner_fraction_p = np.asarray(fraction)
    pd.critical_kind = "smooth" if s == 0 else "one-sided onset at the imposed shape threshold"


def calculate_irods_scale(pd, s=0, PhiP_max=1, *, D=None, progress=None, verbose=None):
    s = _real_scalar(s, "s")
    D = 2 * pd.Rpr_in if D is None else D
    base = IsotropicRodEOS(D, 0, pd.Rpol_in, pd.solvent, nu=getattr(pd, "nu", None))
    PhiP_max = _real_scalar(PhiP_max, "PhiP_max", positive=True)
    if pd.N_steps < 2:
        raise ValueError("N_steps must be at least 2.")
    progress = pd.progress if progress is None else progress
    verbose = pd.verbose if verbose is None else verbose
    phi_c, phi_R_c = _critical_point(base, pd.guess_c)
    eos = ScaledIsotropicRodEOS(D, pd.Rpol_in, pd.solvent, s, phi_c, nu=getattr(pd, "nu", None))
    if PhiP_max < phi_R_c:
        raise ValueError(f"PhiP_max must be >= the reference isotropic critical reservoir concentration {phi_R_c:.10g}.")
    if s == 0:
        calculate_rods(pd, PhiP_max, D=D, L=0, progress=progress, verbose=verbose)
        _install_shape_metadata(pd, s, phi_c, [phi_c, phi_R_c],
                                pd.Pr_Drop_p / pd.phix_bsa, pd.Pr_Sup_p / pd.phix_bsa,
                                np.zeros(pd.N_steps, bool), np.ones(pd.N_steps))
        return pd

    # Use zero common ideal-gas constant for numerical solving. Add the same
    # constant to both phase chemical potentials only when storing results.
    vbsa = eos.volume * 6.022e23
    vpeg = base.polymer_volume * 6.022e23
    phix_bsa, phix_peg = 1 / vbsa, 1 / vpeg
    T = 273.15 + pd.TC
    Lam = 6.626e-34 / np.sqrt(2 * np.pi * pd.mbsa * 1000 * 1.38e-23 * T / 6.022e23)
    mu_constant = np.log(Lam**3 / vbsa)
    pressure_units = phix_bsa * 8.314 * T
    result = {name: np.empty(pd.N_steps) for name in pd.RETURN_FIELDS
              if name not in ("cp", "tp", "cp_list", "tp_list", "params")}
    phi_drop, phi_sup = np.empty(pd.N_steps), np.empty(pd.N_steps)
    boundary, fractions = np.empty(pd.N_steps, bool), np.empty(pd.N_steps)
    previous = None
    if verbose:
        print(f"Scaled isotropic rods: D={D * 1e9:g} nm, s={s:g}; reference phi_c={phi_c:.8g}, onset phi_R={phi_R_c:.8g}")
    for k, phi_R in enumerate(tqdm(np.linspace(phi_R_c, PhiP_max, pd.N_steps),
                                  desc="Scaled isotropic rod binodal", disable=not progress)):
        reservoir = eos.at(phi_R)
        if k == 0 or phi_R == phi_R_c:
            eta_g, eta_d, at_corner, fraction = phi_c, phi_c, True, 0.0
        else:
            eta_g, eta_d, at_corner, fraction = _coexistence(reservoir, previous)
        previous = (eta_g, eta_d)
        j = pd.N_steps - 1 - k
        result["phi_R"][j] = phi_R
        phi_sup[j], phi_drop[j] = eta_g, float(eos.phi_from_eta(eta_d))
        boundary[j], fractions[j] = at_corner, fraction
        for phase, eta in (("Sup", eta_g), ("Drop", eta_d)):
            if phase == "Sup":
                _, mu, p = reservoir.gas.evaluate(eta)
                _, mu0, p0 = base.reference(eta)
                alpha, g, h = reservoir.gas.partition(eta)
            elif at_corner:
                # The free-energy value is the same on the two sides. This
                # convex combination selects the supporting corner slope.
                _, mu, p = ((1 - fraction) * reservoir.gas.evaluate(phi_c)
                            + fraction * reservoir.evaluate(phi_c))
                _, mu0, p0 = ((1 - fraction) * base.reference(phi_c)
                              + fraction * reservoir.reference(phi_c))
                alpha, g, h = ((1 - fraction) * reservoir.gas.partition(phi_c)
                               + fraction * reservoir.partition(phi_c))
            else:
                _, mu, p = reservoir.evaluate(eta)
                _, mu0, p0 = reservoir.reference(eta)
                alpha, g, h = reservoir.partition(eta)
            for key, value in {
                f"Pr_{phase}_p": eta * phix_bsa,
                f"Pol_{phase}_p": alpha * phi_R * phix_peg,
                f"a_{phase}_p": alpha,
                f"qx_{phase}_p": reservoir.qx,
                f"dPidPhiP_{phase}_p": reservoir.dpi,
                f"Pi_{phase}": p * pressure_units,
                f"P0_{phase}": p0 * pressure_units,
                f"Pp_{phase}": (p - p0) * pressure_units,
                f"Mu_{phase}": mu + mu_constant,
                f"Mu0_{phase}": mu0 + mu_constant,
                f"MuR_{phase}": mu - mu0,
                f"g_{phase}": g,
                f"h_{phase}": h,
            }.items():
                result[key][j] = value

    pd._clear_scaled_rod_metadata()
    pd.D, pd.L, pd.aspect_ratio = float(D), 0.0, 1.0
    pd.q, pd.vbsa, pd.vpeg = base.q, vbsa, vpeg
    pd.phix_bsa, pd.phix_peg = phix_bsa, phix_peg
    pd.PhiP_min, pd.PhiP_max = phi_R_c, PhiP_max
    pd.outguess_c, pd.outguess_tp = [phi_c, phi_R_c], None
    pd.q_Drop_p = np.full(pd.N_steps, base.q)
    pd.q_Sup_p = np.full(pd.N_steps, base.q)
    pd.depletion_model = "diameter-based spherical thickness approximation" if pd.solvent in ("good", "theta", "RG") else "ideal penetrable spheres"
    _install_shape_metadata(pd, s, phi_c, [phi_c, phi_R_c], phi_drop, phi_sup, boundary, fractions)
    pd.onset = [phi_c * phix_bsa, base.at(phi_R_c).partition(phi_c)[0] * phi_R_c * phix_peg]
    result.update(cp=None, tp=None, cp_list=[], tp_list=[], params=pd._build_params())
    pd._assign_return_values([result[name] for name in pd.RETURN_FIELDS])
    return pd
