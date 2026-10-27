"""Electrostatic GFVT of Martens et al., J. Colloid Interface Sci. 727, 141573.

https://doi.org/10.1016/j.jcis.2026.141573, Eqs. (1)--(13).
The physical sphere volume sets all concentrations and pressure units; only
the protein reference and the cubic insertion term use phi_e = m*phi.
The paper specifies fluid equilibria, so no crystal/triple point is inferred.
"""

import numpy as np
from scipy.constants import Avogadro
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares
from tqdm import tqdm

from ._quadrature import _T, _W


def _scalar(value, name, *, minimum=None, positive=False):
    if (not np.isscalar(value) or not np.isreal(value)
            or not np.isfinite(value)):
        raise ValueError(f"{name} must be a finite real scalar.")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}.")
    return value


def barker_henderson(R, Z, ionic_strength=None, bjerrum_length=0.71e-9):
    """Return (m, kappa, d_eff) from Eqs. (1), (2), (7).

    R, bjerrum_length and d_eff are in metres; kappa is in inverse metres.
    ionic_strength is the TOTAL ionic strength (salt plus buffer) in mol/L.
    Z is the signed effective charge in elementary-charge units. The default
    Bjerrum length approximates water at room temperature; supply it explicitly
    for a different solvent or temperature. Unscreened nonzero Z is excluded
    because its Barker--Henderson diameter diverges.
    """
    R = _scalar(R, "R", positive=True)
    Z = _scalar(Z, "Z")
    lb = _scalar(bjerrum_length, "bjerrum_length", positive=True)
    if ionic_strength is None:
        if Z != 0:
            raise ValueError("ionic_strength in mol/L is required when Z is nonzero.")
        ionic_strength = 0.0
    strength = _scalar(ionic_strength, "ionic_strength", minimum=0)
    kappa = np.sqrt(8 * np.pi * lb * (1000 * Avogadro * strength))
    if Z == 0:
        return 1.0, kappa, 2 * R
    if strength == 0:
        raise ValueError("ionic_strength must be positive for a screened nonzero charge.")

    # t=kappa*(r-2R) avoids integrating nanometre lengths and evaluating
    # exp(kappa*R). beta*u = contact*exp(-t)/(1+t/(2*kappa*R)).
    kR = kappa * R
    contact = Z**2 * lb / (2 * R * (1 + kR)**2)
    integral, _ = quad(
        lambda t: -np.expm1(-contact * np.exp(-t) / (1 + t / (2 * kR))),
        0, np.inf, epsabs=1e-12, epsrel=1e-11,
    )
    ratio = 1 + integral / (2 * kR)
    m = _scalar(ratio**3, "Barker-Henderson m", minimum=1)
    return m, kappa, 2 * R * ratio


class ElectrostaticEOS:
    """Paper fluid EOS, normalized by the physical protein volume.

    evaluate/reference return (free energy density, chemical potential,
    pressure), in kT/v, kT, kT/v respectively. depletion_exponent=1.53 follows
    Eq. (9); 1.54 permits an exact m=1 comparison with the existing SphereEOS.
    Only the paper's good-solvent polymer closure is implemented here.
    """

    def __init__(self, q, m, mu_constant=0, depletion_exponent=None, *, nu=None,
                 kuhn_crossover=None, protein_radius=None):
        from .polymer import PolymerScaling
        self.polymer = PolymerScaling("good", nu, kuhn_crossover, protein_radius)
        self.q = _scalar(q, "q", positive=True)
        self.m = _scalar(m, "m", minimum=1)
        self.mu_constant = _scalar(mu_constant, "mu_constant")
        if depletion_exponent is None:
            depletion_exponent = 1.53 if nu is None else self.polymer.depletion_power
        self.depletion_exponent = _scalar(depletion_exponent, "depletion_exponent", positive=True)
        if nu is not None and not np.isclose(self.depletion_exponent, self.polymer.depletion_power,
                                            rtol=1e-12, atol=0):
            raise ValueError("With explicit nu, depletion_exponent must equal 2*nu/(3*nu-1).")
        self.phi_ceiling = (1 - 1e-8) / self.m

    def depletion(self, phi_R):
        return (self.polymer.depletion(self.q, phi_R, exponent=self.depletion_exponent),
                self.polymer.pressure_derivative(phi_R) / self.q**3)

    def reference(self, phi):
        e = self.m * phi
        gap = 1 - e
        # Eq. (5): F0 = F_CS(m*phi)/m. In particular, Pi0 is divided by m,
        # while mu0 is not. The ln(m) term follows the printed ideal term.
        f = phi * (self.mu_constant + np.log(e) - 1 + (4 * e - 3 * e**2) / gap**2)
        mu = self.mu_constant + np.log(e) + (3 - e) / gap**3 - 3
        pressure = phi * (1 + e + e**2 - e**3) / gap**3
        return np.array([f, mu, pressure])

    def reference_slopes(self, phi):
        """First and second pressure derivatives with respect to physical phi."""
        e = self.m * phi
        return np.array([
            (1 + 4 * e + 4 * e**2 - 4 * e**3 + e**4) / (1 - e)**4,
            self.m * 4 * (2 + 5 * e - e**2) / (1 - e)**5,
        ])

    def alpha_terms(self, phi, qx, order=3):
        """Eq. (8) and its analytic physical-density derivatives through third.

        Keep x=phi/(1-phi) distinct from y=m*phi/(1-m*phi). Replacing phi
        everywhere by m*phi is not the polymer free volume in the paper.
        """
        gap, egap = 1 - phi, 1 - self.m * phi
        x, y = phi / gap, self.m * phi / egap
        a, b, c = 3 * qx + 3 * qx**2, 4.5 * qx**2, qx**3
        Q = a * x + b * x**2 + c * (y + 3 * y**2 + 3 * y**3)
        alpha = gap * np.exp(-Q)
        if order == 0:
            return alpha
        x1, y1 = gap**-2, self.m * egap**-2
        fy = 1 + 6 * y + 9 * y**2
        Q1 = (a + 2 * b * x) * x1 + c * fy * y1
        L1 = -1 / gap - Q1
        alpha1 = alpha * L1
        if order == 1:
            return alpha, alpha1
        x2, y2 = 2 / gap**3, 2 * self.m**2 / egap**3
        fyy = 6 + 18 * y
        Q2 = 2 * b * x1**2 + (a + 2 * b * x) * x2 + c * (fyy * y1**2 + fy * y2)
        L2 = -1 / gap**2 - Q2
        x3, y3 = 6 / gap**4, 6 * self.m**3 / egap**4
        Q3 = (6 * b * x1 * x2 + (a + 2 * b * x) * x3
              + c * (18 * y1**3 + 3 * fyy * y1 * y2 + fy * y3))
        L3 = -2 / gap**3 - Q3
        return alpha, alpha1, alpha * (L1**2 + L2), alpha * (L1**3 + 3 * L1 * L2 + L3)

    def at(self, phi_R):
        return _ElectrostaticReservoir(self, phi_R)


class _ElectrostaticReservoir:
    def __init__(self, eos, phi_R):
        self.eos = eos
        self.phi_R = _scalar(phi_R, "phi_R", minimum=0)
        grid = self.phi_R * _T**4
        self.grid_qx, dpi = eos.depletion(grid)
        self.weights = _W * (4 * self.phi_R * _T**3) * dpi
        self.qx, self.dpi = eos.depletion(self.phi_R)

    def evaluate(self, phi):
        alpha, alpha1 = self.eos.alpha_terms(phi, self.grid_qx, order=1)
        return self.eos.reference(phi) + np.array([
            -np.dot(self.weights, alpha), -np.dot(self.weights, alpha1),
            np.dot(self.weights, alpha - phi * alpha1),
        ])

    def slopes(self, phi):
        _, _, alpha2, alpha3 = self.eos.alpha_terms(phi, self.grid_qx)
        return self.eos.reference_slopes(phi) - np.array([
            phi * np.dot(self.weights, alpha2),
            np.dot(self.weights, alpha2 + phi * alpha3),
        ])

    def partition(self, phi):
        alpha, alpha1 = self.eos.alpha_terms(phi, self.qx, order=1)
        return np.array([alpha, -alpha1, alpha - phi * alpha1])


def _critical_point(eos, guess):
    def residual(log_values):
        phi, phi_R = np.exp(log_values)
        return eos.at(phi_R).slopes(phi) / eos.reference_slopes(phi)

    lower = np.log([1e-8 / eos.m, 1e-8])
    upper = np.log([0.95 / eos.m, 500.0])
    seeds = [guess, (0.15 / eos.m, 1), (0.1 / eos.m, 10), (0.03 / eos.m, 30)]
    for seed in seeds:
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (2,) or np.any(~np.isfinite(seed)) or np.any(seed <= 0):
            continue
        sol = least_squares(
            residual, np.clip(np.log(seed), lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper), gtol=1e-12, ftol=1e-12, xtol=1e-12, max_nfev=300,
        )
        if sol.success and not np.any(sol.active_mask) and np.max(np.abs(sol.fun)) < 1e-9:
            phi_c, phi_R_c = np.exp(sol.x)
            if (eos.at(phi_R_c * (1 - 1e-4)).slopes(phi_c)[0] > 0
                    and eos.at(phi_R_c * (1 + 1e-4)).slopes(phi_c)[0] < 0):
                return float(phi_c), float(phi_R_c)
    raise RuntimeError("Could not locate an electrostatic fluid critical point (searched 0 < phi_R < 500).")


def _solve_pair(state, phi_c, previous):
    floor, ceiling = np.finfo(float).tiny, state.eos.phi_ceiling
    slope = lambda phi: state.slopes(phi)[0]
    center = phi_c
    if slope(center) >= 0:
        candidates = np.linspace(1e-7 / state.eos.m, 0.95 / state.eos.m, 200)
        center = min(candidates, key=slope)
    if slope(center) >= 0:
        raise RuntimeError(f"Could not bracket electrostatic spinodals at phi_R={state.phi_R:.10g}.")
    spin_g = brentq(slope, floor, center, xtol=1e-15 / state.eos.m)
    spin_l = brentq(slope, center, ceiling, xtol=1e-15 / state.eos.m)
    if spin_g >= spin_l:
        raise RuntimeError("Unresolved electrostatic spinodal separation.")
    midpoint = (spin_g + spin_l) / 2
    half_gap = np.sqrt(3) * (spin_l - spin_g) / 2
    seed = [max(midpoint - half_gap, spin_g / 2), min(midpoint + half_gap, (ceiling + spin_l) / 2)]
    lower, upper = np.log([floor, spin_l]), np.log([spin_g, ceiling])

    def residual(log_phi):
        gas, liquid = np.exp(log_phi)
        return (state.evaluate(gas)[1:] - state.evaluate(liquid)[1:]) / (liquid - gas)

    for guess in (previous, seed):
        if guess is None:
            continue
        sol = least_squares(
            residual, np.clip(np.log(guess), lower + 1e-10, upper - 1e-10),
            bounds=(lower, upper), x_scale="jac", gtol=1e-12, ftol=1e-12,
            xtol=1e-12, max_nfev=200,
        )
        if sol.success and not np.any(sol.active_mask) and np.max(np.abs(sol.fun)) < 1e-8:
            return np.exp(sol.x)
    raise RuntimeError(f"Could not find distinct electrostatic fluids at phi_R={state.phi_R:.10g}.")


def calculate_es(pd, PhiP_max, *, Z=None, ionic_strength=None, m=None,
                 bjerrum_length=0.71e-9, depletion_exponent=None,
                 critical=None, progress=None, verbose=None):
    """Calculate locally; publish a complete PD result only after success."""
    if pd.solvent != "good":
        raise ValueError("GFVT_es implements the paper's solvent='good' polymer model.")
    R = _scalar(pd.Rpr_in, "Rpr_in", positive=True)
    Rp = _scalar(pd.Rpol_in, "Rpol_in", positive=True)
    PhiP_max = _scalar(PhiP_max, "PhiP_max", positive=True)
    if pd.N_steps < 2:
        raise ValueError("N_steps must be at least 2 to include the critical point and upper limit.")
    T = _scalar(273.15 + pd.TC, "temperature in kelvin", positive=True)
    _scalar(pd.mbsa, "mbsa", positive=True)
    lb = _scalar(bjerrum_length, "bjerrum_length", positive=True)
    if m is None:
        if Z is None:
            raise ValueError("Supply either m, or Z with ionic_strength in mol/L.")
        m, kappa, d_eff = barker_henderson(R, Z, ionic_strength, lb)
    else:
        if Z is not None or ionic_strength is not None:
            raise ValueError("Supply m or Z/ionic_strength, not both parameterizations.")
        m = _scalar(m, "m", minimum=1)
        kappa, d_eff = None, 2 * R * m**(1 / 3)
    critical = pd.critical if critical is None else critical
    progress = pd.progress if progress is None else progress
    verbose = pd.verbose if verbose is None else verbose
    vbsa, vpeg = 4 / 3 * np.pi * R**3 * 6.022e23, 4 / 3 * np.pi * Rp**3 * 6.022e23
    phix_bsa, phix_peg = 1 / vbsa, 1 / vpeg
    # Retain PD's additive chemical-potential convention and physical units.
    Lam = 6.626e-34 / np.sqrt(2 * np.pi * pd.mbsa * 1000 * 1.38e-23 * T / 6.022e23)
    eos = ElectrostaticEOS(Rp / R, m, np.log(Lam**3 / vbsa), depletion_exponent,
                          nu=getattr(pd, "nu", None),
                          kuhn_crossover=getattr(pd, "kuhn_crossover", None), protein_radius=R)
    phi_c, phi_R_c = _critical_point(eos, pd.guess_c)
    if PhiP_max < phi_R_c:
        raise ValueError(f"PhiP_max={PhiP_max:g} must be >= the electrostatic critical reservoir concentration {phi_R_c:.10g}.")
    if verbose:
        print(f"GFVT_es: q={eos.q:.6g}, m={m:.6g}, d_eff={d_eff * 1e9:.6g} nm; "
              f"critical phi={phi_c:.8g}, phi_R={phi_R_c:.8g}")

    result = {name: np.empty(pd.N_steps) for name in pd.RETURN_FIELDS
              if name not in ("cp", "tp", "cp_list", "tp_list", "params")}
    current, pair = phi_R_c, None
    seed_step = max(phi_R_c, 1e-3) * 0.01
    min_step = max(phi_R_c, 1.0) * 1e-10
    pressure_units = phix_bsa * 8.314 * T
    for k, target in enumerate(tqdm(np.linspace(phi_R_c, PhiP_max, pd.N_steps),
                                    desc="Electrostatic GFVT binodal", disable=not progress)):
        attempts = 0
        while current < target:
            step = min(target - current, max(seed_step, (current - phi_R_c) / 2))
            while True:
                attempts += 1
                if attempts > 100:
                    raise RuntimeError(f"Electrostatic continuation stalled near phi_R={current:.10g}.")
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
        state = eos.at(target)
        for phase, phi in (("Sup", gas), ("Drop", liquid)):
            _, mu, pressure = state.evaluate(phi)
            _, mu0, p0 = eos.reference(phi)
            alpha, g, h = state.partition(phi)
            for key, value in {
                f"Pr_{phase}_p": phi * phix_bsa,
                f"Pol_{phase}_p": alpha * target * phix_peg,
                f"a_{phase}_p": alpha,
                f"qx_{phase}_p": state.qx,
                f"dPidPhiP_{phase}_p": state.dpi,
                f"Pi_{phase}": pressure * pressure_units,
                f"P0_{phase}": p0 * pressure_units,
                f"Pp_{phase}": (pressure - p0) * pressure_units,
                f"Mu_{phase}": mu,
                f"Mu0_{phase}": mu0,
                f"MuR_{phase}": mu - mu0,
                f"g_{phase}": g,
                f"h_{phase}": h,
            }.items():
                result[key][j] = value

    cp_list = []
    if critical:
        guess = (phi_c, phi_R_c)
        for q in tqdm(np.linspace(10, 0.388, pd.N_steps),
                      desc="Electrostatic critical line", disable=not progress):
            scan_eos = ElectrostaticEOS(q, m, eos.mu_constant, depletion_exponent,
                                       nu=getattr(pd, "nu", None),
                                       kuhn_crossover=getattr(pd, "kuhn_crossover", None),
                                       protein_radius=R)
            guess = _critical_point(scan_eos, guess)
            pc, rc = guess
            scale = 1 / (4 / 3 * np.pi * (q * R)**3 * 6.022e23)
            cp_list.append([q, R, q * R, scale, pc, rc, pc * phix_bsa,
                            scan_eos.at(rc).partition(pc)[0] * rc * scale])
        cp_list = np.asarray(cp_list)

    pd._clear_scaled_rod_metadata()
    for name in ("D", "L", "aspect_ratio", "depletion_model"):
        pd.__dict__.pop(name, None)
    pd.geometry, pd.model_kind = "spheres", "electrostatic"
    pd.es_m, pd.es_Z, pd.es_ionic_strength = m, Z, ionic_strength
    pd.es_bjerrum_length, pd.es_kappa, pd.es_d_eff = lb, kappa, d_eff
    pd.es_depletion_exponent = eos.depletion_exponent
    pd.q, pd.vbsa, pd.vpeg = eos.q, vbsa, vpeg
    pd.phix_bsa, pd.phix_peg = phix_bsa, phix_peg
    pd.PhiP_min, pd.PhiP_max = phi_R_c, PhiP_max
    pd.outguess_c, pd.outguess_tp = [phi_c, phi_R_c], None
    pd.s = 0
    pd.q_Drop_p = np.full(pd.N_steps, eos.q)
    pd.q_Sup_p = np.full(pd.N_steps, eos.q)
    result.update(cp=[phi_c * phix_bsa, eos.at(phi_R_c).partition(phi_c)[0] * phi_R_c * phix_peg],
                  tp=None, cp_list=cp_list, tp_list=[], params=pd._build_params())
    pd._assign_return_values([result[name] for name in pd.RETURN_FIELDS])
    return pd
