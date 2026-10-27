"""Phase diagram object for GFVT calculations.

``PD.GFVT`` uses the notebook thermodynamics with continuation along distinct
fluid branches. Results are exposed as named attributes and can still be
converted back to the legacy list format.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.stats import linregress
from tqdm import tqdm

from .polymer import PolymerScaling, KuhnCrossover


PARAM_FIELDS = (
    "Rpr_in",
    "Rpol_in",
    "mbsa",
    "mpeg",
    "solvent",
    "N_steps",
    "PhiP_min",
    "PhiP_max",
    "outguess_c",
    "outguess_tp",
    "guess_b",
    "TC",
    "phix_bsa",
    "phix_peg",
)

RETURN_FIELDS = (
    "phi_R",
    "Pr_Drop_p",
    "Pol_Drop_p",
    "a_Drop_p",
    "Pr_Sup_p",
    "Pol_Sup_p",
    "a_Sup_p",
    "qx_Drop_p",
    "dPidPhiP_Drop_p",
    "qx_Sup_p",
    "dPidPhiP_Sup_p",
    "Pi_Drop",
    "P0_Drop",
    "Pp_Drop",
    "Mu_Drop",
    "Mu0_Drop",
    "MuR_Drop",
    "Pi_Sup",
    "P0_Sup",
    "Pp_Sup",
    "Mu_Sup",
    "Mu0_Sup",
    "MuR_Sup",
    "cp",
    "cp_list",
    "tp",
    "tp_list",
    "g_Drop",
    "g_Sup",
    "h_Drop",
    "h_Sup",
    "params",
)


def _as_gfvt_values(fvt):
    if hasattr(fvt, "to_legacy_list"):
        return fvt.to_legacy_list()
    return fvt


def gfvt_unpack(fvt):
    """Return named GFVT arrays from a legacy return list or ``PD`` object."""
    values = _as_gfvt_values(fvt)
    if len(values) != len(RETURN_FIELDS):
        raise ValueError(f"Expected {len(RETURN_FIELDS)} GFVT values, got {len(values)}.")

    return {
        "phi_R": np.asarray(values[0], dtype=float),
        "Pr_Drop": np.asarray(values[1], dtype=float),
        "Pol_Drop": np.asarray(values[2], dtype=float),
        "alpha_Drop": np.asarray(values[3], dtype=float),
        "Pr_Sup": np.asarray(values[4], dtype=float),
        "Pol_Sup": np.asarray(values[5], dtype=float),
        "alpha_Sup": np.asarray(values[6], dtype=float),
        "qx_Drop": np.asarray(values[7], dtype=float),
        "dPidPhiP_Drop": np.asarray(values[8], dtype=float),
        "qx_Sup": np.asarray(values[9], dtype=float),
        "dPidPhiP_Sup": np.asarray(values[10], dtype=float),
        "params": values[-1],
        "nu": getattr(fvt, "nu", None),
        "s": getattr(fvt, "s", 0),
    }


def qxf_external(q, phi_R, solvent, *, nu=None, kuhn_crossover=None, protein_radius=None):
    """GFVT reservoir thickness and pressure derivative with shared scaling."""
    return PolymerScaling(solvent, nu, kuhn_crossover, protein_radius).qxf(q, phi_R)


def alpha_spt(eta, q_eff):
    """Scaled-particle free-volume fraction for hard spheres."""
    return np.exp(_log_alpha_spt(eta, q_eff))


def _log_alpha_spt(eta, q_eff):
    eta = np.asarray(eta, dtype=float)
    y = eta / (1.0 - eta)
    a = 3.0 * q_eff + 3.0 * q_eff**2 + q_eff**3
    b = 4.5 * q_eff**2 + 3.0 * q_eff**3
    c = 3.0 * q_eff**3
    return np.log1p(-eta) - (a * y + b * y**2 + c * y**3)


def solve_q_app_from_ratio(eta_sup, eta_drop, pol_sup, pol_drop, q_bounds=(1e-6, 100.0)):
    """Solve apparent q from one tie line using PEG partitioning."""
    if not (
        np.isfinite(eta_sup)
        and np.isfinite(eta_drop)
        and np.isfinite(pol_sup)
        and np.isfinite(pol_drop)
    ):
        return np.nan
    if pol_sup <= 0 or pol_drop <= 0:
        return np.nan
    if not (0 <= eta_sup < 1 and 0 <= eta_drop < 1):
        return np.nan

    target = np.log(pol_drop / pol_sup)

    def residual(q_eff):
        return _log_alpha_spt(eta_drop, q_eff) - _log_alpha_spt(eta_sup, q_eff) - target

    lo, hi = q_bounds
    try:
        f_lo = residual(lo)
        f_hi = residual(hi)
        if f_lo == 0:
            return lo
        if f_hi == 0:
            return hi
        if np.sign(f_lo) != np.sign(f_hi):
            return brentq(residual, lo, hi, maxiter=200)
    except Exception:
        pass

    try:
        opt = minimize_scalar(lambda x: residual(x) ** 2, bounds=q_bounds, method="bounded")
        return opt.x if opt.success else np.nan
    except Exception:
        return np.nan


def add_q_app_to_gfvt(fvt):
    """Return q_app and tie-line variables for a GFVT result."""
    u = gfvt_unpack(fvt)
    params = u["params"]
    Rpr, Rpol, solvent = params[0], params[1], params[4]
    phix_bsa = params[-2]
    q_input = Rpol / Rpr

    eta_sup = u["Pr_Sup"] / phix_bsa
    eta_drop = u["Pr_Drop"] / phix_bsa
    q_app = np.array(
        [
            solve_q_app_from_ratio(es, ed, ps, pd)
            for es, ed, ps, pd in zip(eta_sup, eta_drop, u["Pol_Sup"], u["Pol_Drop"])
        ]
    )

    if getattr(fvt, "model_kind", None) == "density_scaled_nu":
        q_eff_reservoir = np.asarray(u["qx_Sup"], dtype=float)
        dPidPhiP = np.asarray(u["dPidPhiP_Sup"], dtype=float)
    elif getattr(fvt, "model_kind", None) == "electrostatic" and getattr(fvt, "kuhn_crossover", None) is not None:
        q_eff_reservoir, dPidPhiP = fvt._qxf(q_input, u["phi_R"])
    else:
        q_eff_reservoir, dPidPhiP = qxf_external(
            q_input, u["phi_R"], solvent, nu=u["nu"],
            kuhn_crossover=getattr(fvt, "kuhn_crossover", None), protein_radius=Rpr)
    return {
        "phi_R": u["phi_R"],
        "eta_sup": eta_sup,
        "eta_drop": eta_drop,
        "Pr_Sup_mM": u["Pr_Sup"],
        "Pr_Drop_mM": u["Pr_Drop"],
        "Pol_Sup_mM": u["Pol_Sup"],
        "Pol_Drop_mM": u["Pol_Drop"],
        "alpha_Sup": u["alpha_Sup"],
        "alpha_Drop": u["alpha_Drop"],
        "q_app": q_app,
        "q_eff_reservoir": q_eff_reservoir,
        "dPidPhiP": dPidPhiP,
        "q_input": q_input,
        "solvent": solvent,
    }


def add_X_ins_to_gfvt(fvt):
    """Calculate X_ins along the GFVT reservoir-polymer grid."""
    u = gfvt_unpack(fvt)
    params = u["params"]
    Rpr, Rpol, solvent = params[0], params[1], params[4]
    q_input = Rpol / Rpr

    phi_R = np.asarray(u["phi_R"], dtype=float)
    order = np.argsort(phi_R)
    phi_sorted = phi_R[order]

    if getattr(fvt, "model_kind", None) == "density_scaled_nu":
        q_eff = np.asarray(u["qx_Sup"], dtype=float)[order]
        dPidPhiP = np.asarray(u["dPidPhiP_Sup"], dtype=float)[order]
    elif getattr(fvt, "model_kind", None) == "electrostatic" and getattr(fvt, "kuhn_crossover", None) is not None:
        q_eff, dPidPhiP = fvt._qxf(q_input, phi_sorted)
    else:
        q_eff, dPidPhiP = qxf_external(
            q_input, phi_sorted, solvent, nu=u["nu"],
            kuhn_crossover=getattr(fvt, "kuhn_crossover", None), protein_radius=Rpr)
    integrand = (1.0 + q_eff) ** 3 * dPidPhiP
    X_sorted = cumulative_trapezoid(integrand, phi_sorted, initial=0.0)

    X_ins = np.empty_like(X_sorted)
    X_ins[order] = X_sorted
    X_integrand = np.empty_like(integrand)
    X_integrand[order] = integrand

    out = add_q_app_to_gfvt(fvt)
    out["X_ins"] = X_ins
    out["X_integrand"] = X_integrand
    return out


def plot_q_app(fvt, label=None, ax=None):
    """Plot apparent q from GFVT tie lines against reservoir polymer coordinate."""
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(6, 4))

    d = add_q_app_to_gfvt(fvt)
    ax.plot(d["phi_R"], d["q_app"], "o", label=f"{label} q_app" if label else "q_app")
    ax.plot(
        d["phi_R"],
        d["q_eff_reservoir"],
        "-",
        label=f"{label} q_eff" if label else "q_eff",
    )
    ax.set_xlabel(r"Reservoir $\Phi_p^R = c_p^R/c_p^*$")
    ax.set_ylabel(r"$q=\delta_s/R$")
    ax.legend(frameon=False)
    return ax, d


def plot_dilute_arm_vs_Xins(fvt, label=None, ax=None, fit=True, min_c=0.0, max_phi_R=None):
    """Plot log dilute-arm protein concentration against insertion work."""
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(6, 4))

    d = add_X_ins_to_gfvt(fvt)
    X = np.asarray(d["X_ins"], dtype=float)
    c = np.asarray(d["Pr_Sup_mM"], dtype=float)
    phi_R = np.asarray(d["phi_R"], dtype=float)

    mask = np.isfinite(X) & np.isfinite(c) & (c > min_c)
    if max_phi_R is not None:
        mask &= phi_R <= max_phi_R

    X_fit = X[mask]
    y_fit = np.log(c[mask])
    ax.plot(X_fit, y_fit, "o", label=label if label else "GFVT dilute arm")

    fit_result = None
    if fit and len(X_fit) >= 3:
        lr = linregress(X_fit, y_fit)
        A = lr.intercept
        S = -lr.slope
        xline = np.linspace(np.nanmin(X_fit), np.nanmax(X_fit), 200)
        ax.plot(
            xline,
            A - S * xline,
            "-",
            label=f"fit: A={A:.3g}, S={S:.3g}, R^2={lr.rvalue**2:.3f}",
        )
        fit_result = {
            "A": A,
            "S": S,
            "slope": lr.slope,
            "intercept": lr.intercept,
            "r2": lr.rvalue**2,
            "pvalue": lr.pvalue,
            "stderr": lr.stderr,
            "n": len(X_fit),
        }

    ax.set_xlabel(r"Insertion work $X_{ins}$")
    ax.set_ylabel(r"$\ln c_{\mathrm{BSA}}^{sup}$, c in mM")
    ax.legend(frameon=False)
    return ax, d, fit_result


class PD:
    """GFVT phase diagram with notebook-compatible state and outputs.

    nu=0.588 selects consistent good-solvent scaling throughout the model.
    nu=None preserves legacy rounded equations. Rpol_in remains the supplied
    dilute radius of gyration; it is not inferred or changed by nu. Save the
    whole PD object to retain nu (legacy 32-item lists cannot store it).
    kuhn_crossover=KuhnCrossover(b=0.75e-9, m=4, s=0.75) optionally changes
    the full good-solvent depletion closure before curvature mapping. None
    (default) and s=1 retain the baseline. Save PD to retain this metadata.
    """

    PARAM_FIELDS = PARAM_FIELDS
    RETURN_FIELDS = RETURN_FIELDS

    def __init__(
        self,
        Rpr_in: float,
        Rpol_in: float,
        mbsa: float,
        mpeg: float,
        solvent: str,
        N_steps: int,
        guess_c: Iterable[float],
        *,
        guess_b: Iterable[float] = (0.0001, 0.5),
        critical: bool = False,
        TC: float = 22,
        progress: bool = True,
        verbose: bool = True,
        nu: float | None = None,
        kuhn_crossover: KuhnCrossover | None = None,
    ) -> None:
        PolymerScaling(solvent, nu, kuhn_crossover, Rpr_in)
        self.kuhn_crossover = kuhn_crossover
        self.nu = nu
        self.Rpr_in = Rpr_in
        self.Rpol_in = Rpol_in
        self.mbsa = mbsa
        self.mpeg = mpeg
        self.solvent = solvent
        self.N_steps = int(N_steps)
        self.guess_c = list(guess_c)
        self.guess_b = list(guess_b)
        self.critical = critical
        self.TC = TC
        self.progress = progress
        self.verbose = verbose
        self.geometry = "spheres"

        self.vbsa = 4 / 3 * np.pi * self.Rpr_in**3 * 6.022e23
        self.vpeg = 4 / 3 * np.pi * self.Rpol_in**3 * 6.022e23
        self.q = self.Rpol_in / self.Rpr_in
        self.phix_bsa = 1 / self.vbsa
        self.phix_peg = 1 / self.vpeg

        self.PhiP_min = None
        self.PhiP_max = None
        self.outguess_c = None
        self.outguess_tp = None

        for field in RETURN_FIELDS:
            setattr(self, field, None)
        self.params = self._build_params()
        self._sync_aliases()

    @classmethod
    def from_legacy_list(cls, values: list[Any]) -> "PD":
        """Build a ``PD`` object from the notebook's 32-item return list."""
        if len(values) != len(RETURN_FIELDS):
            raise ValueError(
                f"Expected {len(RETURN_FIELDS)} legacy values, got {len(values)}."
            )

        params = values[-1]
        if len(params) != len(PARAM_FIELDS):
            raise ValueError(
                f"Expected {len(PARAM_FIELDS)} params values, got {len(params)}."
            )

        (
            Rpr_in,
            Rpol_in,
            mbsa,
            mpeg,
            solvent,
            N_steps,
            PhiP_min,
            PhiP_max,
            outguess_c,
            outguess_tp,
            guess_b,
            TC,
            phix_bsa,
            phix_peg,
        ) = params

        pd = cls(
            Rpr_in,
            Rpol_in,
            mbsa,
            mpeg,
            solvent,
            N_steps,
            outguess_c,
            guess_b=guess_b,
            TC=TC,
            progress=False,
            verbose=False,
        )
        pd.PhiP_min = PhiP_min
        pd.PhiP_max = PhiP_max
        pd.outguess_c = outguess_c
        pd.outguess_tp = outguess_tp
        pd.phix_bsa = phix_bsa
        pd.phix_peg = phix_peg
        pd._assign_return_values(values)
        return pd

    def GFVT(
        self,
        *,
        PhiP_max_override: float | None = None,
        critical: bool | None = None,
        progress: bool | None = None,
        verbose: bool | None = None,
    ) -> "PD":
        """Calculate fluid coexistence from the critical point toward higher phi_R.

        PhiP_max_override replaces the default triple-point reservoir limit and
        must be finite and at least the critical reservoir concentration. The
        solver uses internal continuation steps and excludes equal-density roots
        away from criticality. Stored arrays retain descending phi_R order.
        Reservoir integrals use cached, transformed Gaussian quadrature shared
        by GFVT_full and GFVT_scale; the sphere equations of state are unchanged.
        """
        return self._calculate_gfvt(
            PhiP_max_override=PhiP_max_override, critical=critical,
            progress=progress, verbose=verbose,
        )

    def _calculate_gfvt(
        self,
        *,
        PhiP_max_override=None,
        critical=None,
        progress=None,
        verbose=None,
        _reference_only=False,
    ):
        """Ordinary sphere GFVT, optionally returning only reference landmarks.

        The exclusion-layer model reuses this critical/triple calculation
        without calculating an extra ordinary binodal.
        """
        critical = self.critical if critical is None else critical
        progress = self.progress if progress is None else progress
        verbose = self.verbose if verbose is None else verbose

        # Rod solves use the complete spherocylinder volume. A subsequent
        # ordinary GFVT calculation must restore the constructor's spheres.
        if getattr(self, "geometry", "spheres") in ("isotropic_rods", "isotropic_rods_scaled", "spherocylinders"):
            self.geometry = "spheres"
            self.vbsa = 4 / 3 * np.pi * self.Rpr_in**3 * 6.022e23
            self.vpeg = 4 / 3 * np.pi * self.Rpol_in**3 * 6.022e23
            self.q = self.Rpol_in / self.Rpr_in
            self.phix_bsa, self.phix_peg = 1 / self.vbsa, 1 / self.vpeg
            for name in ("D", "L", "aspect_ratio", "depletion_model"):
                self.__dict__.pop(name, None)
        self._clear_scaled_rod_metadata()

        Rpr_in = self.Rpr_in
        Rpol_in = self.Rpol_in
        mbsa = self.mbsa
        mpeg = self.mpeg
        solvent = self.solvent
        N_steps = self.N_steps
        guess_c = self.guess_c
        guess_b = self.guess_b

        vbsa = self.vbsa
        vpeg = self.vpeg
        q = self.q

        if verbose:
            print(f"q = {q:.2f}, Rbsa = {Rpr_in * 1e9:.2f} nm, Rp = {Rpol_in * 1e9:.2f} nm")

        phix_peg = self.phix_peg
        phix_bsa = self.phix_bsa

        hp = 6.626e-34
        kb = 1.38e-23
        Nav = 6.022e23
        Rb = 8.314
        TC = self.TC
        T = 273.15 + TC
        Lam = hp / (2 * np.pi * mbsa * 1000 * kb * T / Nav) ** (1 / 2)

        from .spheres import SphereEOS

        mu_constant = np.log(Lam**3 / vbsa)
        nu = getattr(self, "nu", None)
        kuhn = getattr(self, "kuhn_crossover", None)
        eos_options = dict(nu=nu, kuhn_crossover=kuhn, protein_radius=Rpr_in)
        eos = SphereEOS(q, solvent, mu_constant, **eos_options)

        @lru_cache(maxsize=32)
        def reservoir(phi_pol, phase_q=q):
            # A bounded, solve-local cache also serves critical/triple scans.
            # It is released when this calculation returns, not pickled on PD.
            model = eos if phase_q == q else SphereEOS(phase_q, solvent, mu_constant, **eos_options)
            return model.at(phi_pol)

        def alpha(phi, phi_pol, phase_q, solvent):
            return alpha_spt(phi, eos.polymer.qxf(phase_q, phi_pol)[0])

        def triple_residual(values, phase_q):
            state = reservoir(values[3], phase_q)
            gas = state.evaluate(values[0])
            liquid = state.evaluate(values[1])
            solid = state.evaluate(values[2], solid=True)
            return [gas[0] - liquid[0], liquid[0] - solid[0],
                    gas[1] - liquid[1], liquid[1] - solid[1]]

        array_names = (
            "Pr_Drop_p",
            "Pol_Drop_p",
            "a_Drop_p",
            "qx_Drop_p",
            "dPidPhiP_Drop_p",
            "Pr_Sup_p",
            "Pol_Sup_p",
            "a_Sup_p",
            "qx_Sup_p",
            "dPidPhiP_Sup_p",
            "phi_R",
            "P0_Drop",
            "P0_Sup",
            "Pi_Drop",
            "Pi_Sup",
            "Pp_Drop",
            "Pp_Sup",
            "Mu_Drop",
            "Mu_Sup",
            "MuR_Drop",
            "MuR_Sup",
            "Mu0_Drop",
            "Mu0_Sup",
            "g_Drop",
            "g_Sup",
            "h_Drop",
            "h_Sup",
        )
        arrays = {name: np.empty(N_steps) for name in array_names}
        (
            Pr_Drop_p,
            Pol_Drop_p,
            a_Drop_p,
            qx_Drop_p,
            dPidPhiP_Drop_p,
            Pr_Sup_p,
            Pol_Sup_p,
            a_Sup_p,
            qx_Sup_p,
            dPidPhiP_Sup_p,
            phi_R,
            P0_Drop,
            P0_Sup,
            Pi_Drop,
            Pi_Sup,
            Pp_Drop,
            Pp_Sup,
            Mu_Drop,
            Mu_Sup,
            MuR_Drop,
            MuR_Sup,
            Mu0_Drop,
            Mu0_Sup,
            g_Drop,
            g_Sup,
            h_Drop,
            h_Sup,
        ) = (arrays[name] for name in array_names)
        
        cp_list = np.empty((N_steps, 8))
        tp_list = np.empty((0, 14))

        def fcrit(variabs):
            return reservoir(variabs[1]).slopes(variabs[0])

        sol = least_squares(
            fcrit,
            (guess_c[0], guess_c[1]),
            bounds=((1e-16, 1e-16), (1, 500)),
            jac="2-point",
            gtol=3e-16,
            ftol=3e-16,
            xtol=3e-16,
        )

        if (nu is not None or kuhn is not None) and (not sol.success or np.max(np.abs(sol.fun)) > 1e-8):
            raise RuntimeError(f"Could not resolve the critical point at q={q:g}, nu={nu}.")
        crit1 = float(sol.x[0])
        crit2 = float(sol.x[1])
        outguess_c = [crit1, crit2]
        if PhiP_max_override is not None and (
            not np.isfinite(PhiP_max_override) or PhiP_max_override < crit2
        ):
            raise ValueError(
                f"PhiP_max={PhiP_max_override!r} must be finite and >= the critical "
                f"reservoir concentration {crit2:.10g}. There is no fluid-fluid "
                "binodal below this concentration."
            )
        Pr_Crit = crit1 * phix_bsa
        Pol_Crit = alpha(crit1, crit2, q, solvent) * crit2 * phix_peg
        cp = [Pr_Crit, Pol_Crit]

        if verbose:
            print(f"BSA_crit = {crit1:.4f}, PEG_crit = {crit2:.4f}")
            print(f"[BSA]_crit = {Pr_Crit:.2f} mM, [PEG]_crit = {Pol_Crit:.2f} mM\n")

        def ftp(variabs):
            return triple_residual(variabs, q)

        tp_sol = least_squares(
            ftp,
            (1e-4, outguess_c[0] + 0.4, 0.7, outguess_c[1]),
            bounds=((1e-16, outguess_c[0], outguess_c[0], 1e-16), (outguess_c[0], 1, 1, 500)),
            jac="2-point",
            gtol=3e-16,
            ftol=3e-16,
            xtol=3e-16,
        )

        if (nu is not None or kuhn is not None) and (not tp_sol.success or np.max(np.abs(tp_sol.fun)) > 1e-8):
            raise RuntimeError(f"Could not resolve the triple point at q={q:g}, nu={nu}.")
        tp1 = float(tp_sol.x[0])
        tp2 = float(tp_sol.x[1])
        tp3 = float(tp_sol.x[2])
        tp4 = float(tp_sol.x[3])
        outguess_tp = [tp1, tp2, tp3, tp4]
        Pr_tp1 = tp1 * phix_bsa
        Pr_tp2 = tp2 * phix_bsa
        Pr_tp3 = tp3 * phix_bsa
        Pol_tp1 = alpha(tp1, tp4, q, solvent) * tp4 * phix_peg
        Pol_tp2 = alpha(tp2, tp4, q, solvent) * tp4 * phix_peg
        Pol_tp3 = alpha(tp3, tp4, q, solvent) * tp4 * phix_peg
        tp = [Pr_tp1, Pr_tp2, Pr_tp3, Pol_tp1, Pol_tp2, Pol_tp3]

        if verbose:
            print(f"BSA_tp_G = {tp1:.4f}, BSA_tp_L = {tp2:.4f}, BSA_tp_S = {tp3:.4f}, PEG_tp = {tp4:.4f}")
            print(
                f"[BSA]_tp_G = {Pr_tp1:.2f} mM, [BSA]_tp_L = {Pr_tp2:.2f} mM, "
                f"[BSA]_tp_S = {Pr_tp3:.2f} mM, [PEG]_tp_G = {Pol_tp1:.2f} mM, "
                f"[PEG]_tp_L = {Pol_tp2:.2f} mM, [PEG]_tp_S = {Pol_tp3:.2f} mM\n"
            )

        PhiP_min = crit2
        PhiP_max = tp4 if PhiP_max_override is None else PhiP_max_override

        endpoint = None
        if critical:
            if nu is not None or (kuhn is not None and kuhn.s != 1):
                from .spheres import critical_endpoint
                endpoint = critical_endpoint(solvent, **eos_options)
                qrcep = endpoint["q"]
            elif solvent == "good":
                qrcep = 0.388
            elif solvent == "theta":
                qrcep = 0.337
            elif solvent == "RG":
                qrcep = 0.388
            else:
                qrcep = 0.1

            Qlist = np.flip(np.linspace(qrcep, 10, N_steps))
            previous_scan_q = q

            for idx, qr in tqdm(
                enumerate(Qlist),
                desc="Calculating Critical and Triple Lines",
                total=len(Qlist),
                unit="step",
                disable=not progress,
            ):
                Rpol_qr = qr * Rpr_in
                phix_peg_qr = 1 / (4 / 3 * np.pi * Rpol_qr**3 * 6.022e23)
                # Fleer--Tuinier Eq. 6.14 gives phi_R ~ q**(1/gamma).
                # Use it only to predict a starting guess; solve every point
                # again. An unscaled reservoir guess can find a spurious
                # minimum in a coarse triple-line scan.
                reservoir_guess_scale = ((qr / previous_scan_q)**(1/eos.polymer.gamma)
                                         if nu is not None or (kuhn is not None and kuhn.s != 1) else 1.0)

                def fcrit(variabs):
                    return reservoir(variabs[1], qr).slopes(variabs[0])

                if endpoint is not None and qr == qrcep:
                    # This point was already solved with all four endpoint
                    # equations; do not seed it from a distant sampled q.
                    crit1, crit2 = endpoint["phi_fluid"], endpoint["phi_R"]
                else:
                    sol = least_squares(
                        fcrit,
                        (crit1, crit2 * reservoir_guess_scale),
                        bounds=((1e-16, 1e-16), (1, 500)),
                        jac="2-point",
                        gtol=3e-16,
                        ftol=3e-16,
                        xtol=3e-16,
                    )

                    if (nu is not None or kuhn is not None) and (not sol.success or np.max(np.abs(sol.fun)) > 1e-8):
                        raise RuntimeError(f"Could not resolve the critical point at q={qr:g}, nu={nu}.")
                    crit1 = float(sol.x[0])
                    crit2 = float(sol.x[1])
                Pr_Crit_t = crit1 * phix_bsa
                Pol_Crit_t = alpha(crit1, crit2, qr, solvent) * crit2 * phix_peg_qr
                cp_list[idx] = [qr, Rpr_in, Rpol_qr, phix_peg_qr, crit1, crit2, Pr_Crit_t, Pol_Crit_t]

                if qr > qrcep:

                    def ftp(variabs):
                        return triple_residual(variabs, qr)

                    tp_sol = least_squares(
                        ftp,
                        (tp1, tp2, tp3, tp4 * reservoir_guess_scale),
                        bounds=((1e-16, crit1, crit1, 1e-16), (crit1, 1, 1, 500)),
                        jac="2-point",
                        gtol=3e-16,
                        ftol=3e-16,
                        xtol=3e-16,
                    )

                    if (nu is not None or kuhn is not None) and (not tp_sol.success or np.max(np.abs(tp_sol.fun)) > 1e-8):
                        raise RuntimeError(f"Could not resolve the triple point at q={qr:g}, nu={nu}.")
                    tp1 = float(tp_sol.x[0])
                    tp2 = float(tp_sol.x[1])
                    tp3 = float(tp_sol.x[2])
                    tp4 = float(tp_sol.x[3])
                    Pr_tp1 = tp1 * phix_bsa
                    Pr_tp2 = tp2 * phix_bsa
                    Pr_tp3 = tp3 * phix_bsa
                    Pol_tp1 = alpha(tp1, tp4, qr, solvent) * tp4 * phix_peg_qr
                    Pol_tp2 = alpha(tp2, tp4, qr, solvent) * tp4 * phix_peg_qr
                    Pol_tp3 = alpha(tp3, tp4, qr, solvent) * tp4 * phix_peg_qr
                    tp_list = np.vstack(
                        [
                            tp_list,
                            [
                                qr,
                                Rpr_in,
                                Rpol_qr,
                                phix_peg_qr,
                                tp1,
                                tp2,
                                tp3,
                                tp4,
                                Pr_tp1,
                                Pr_tp2,
                                Pr_tp3,
                                Pol_tp1,
                                Pol_tp2,
                                Pol_tp3,
                            ],
                        ]
                    )
                previous_scan_q = qr
        else:
            cp_list = []
            tp_list = []

        if _reference_only:
            return dict(cp=cp, tp=tp, cp_list=cp_list, tp_list=tp_list,
                        outguess_c=outguess_c, outguess_tp=outguess_tp)

        # The optional critical-line scan reuses crit1/crit2 for other q values.
        crit1, crit2 = outguess_c
        PhiP = np.linspace(PhiP_min, PhiP_max, N_steps)
        phi_floor = np.finfo(float).tiny
        phi_ceiling = 1 - 1e-8
        q_drop_values = np.empty(N_steps)

        def solve_pair(phi_pol, previous):
            """Keep the two fluids outside their respective spinodal boundaries."""
            state = reservoir(phi_pol)

            def slope(phi):
                return state.slopes(phi)[0]

            try:
                spin_gas = brentq(slope, phi_floor, crit1, xtol=1e-15)
                spin_liquid = brentq(slope, crit1, phi_ceiling, xtol=1e-15)
            except ValueError as exc:
                raise RuntimeError(
                    f"Could not bracket the fluid spinodals at phi_R={phi_pol:.10g}."
                ) from exc
            if spin_gas >= spin_liquid:
                raise RuntimeError(f"Unresolved spinodal separation at phi_R={phi_pol:.10g}.")

            lower = np.log([phi_floor, spin_liquid])
            upper = np.log([spin_gas, phi_ceiling])
            midpoint = (spin_gas + spin_liquid) / 2
            half_gap = np.sqrt(3) * (spin_liquid - spin_gas) / 2
            seed = [
                max(midpoint - half_gap, spin_gas / 2),
                min(midpoint + half_gap, (1 + spin_liquid) / 2),
            ]

            def residual(log_phi):
                gas, liquid = np.exp(log_phi)
                gap = liquid - gas
                # Removing the common gap factor improves conditioning near
                # criticality, where both unscaled differences become tiny.
                return (state.evaluate(gas) - state.evaluate(liquid)) / gap

            for guess in (previous, seed):
                if guess is None:
                    continue
                initial = np.clip(np.log(guess), lower + 1e-10, upper - 1e-10)
                solution = least_squares(
                    residual, initial, bounds=(lower, upper), x_scale="jac",
                    gtol=1e-13, ftol=1e-13, xtol=1e-13, max_nfev=150,
                )
                if solution.success and np.max(np.abs(solution.fun)) <= 1e-8:
                    return np.exp(solution.x)
            raise RuntimeError(
                f"Could not find distinct fluid phases at phi_R={phi_pol:.10g}; "
                f"coexistence residual={np.max(np.abs(solution.fun)):.3g}."
            )

        current_phi_R = crit2
        current_pair = None
        seed_step = max(crit2, 1e-3) * 0.01
        min_step = max(crit2, 1.0) * 1e-10

        for k, i in enumerate(tqdm(PhiP, desc="Calculating Binodal", unit="step", disable=not progress)):
            # Solve outward from criticality, but preserve the public array order
            # used by molplot and the legacy return list (critical point last).
            j = N_steps - 1 - k
            continuation_attempts = 0
            while current_phi_R < i:
                step = min(i - current_phi_R, max(seed_step, (current_phi_R - crit2) / 2))
                while True:
                    continuation_attempts += 1
                    if continuation_attempts > 100:
                        raise RuntimeError(
                            f"Binodal continuation stalled near phi_R={current_phi_R:.10g} "
                            f"while advancing to {i:.10g}."
                        )
                    target = min(i, current_phi_R + step)
                    try:
                        next_pair = solve_pair(target, current_pair)
                    except RuntimeError:
                        step /= 2
                        if step < min_step or current_phi_R + step == current_phi_R:
                            raise
                    else:
                        current_phi_R = target
                        current_pair = next_pair
                        break

            # Equal phases are physical exactly at the critical endpoint; do not
            # run the degenerate coexistence solve there.
            g1, g2 = (crit1, crit1) if current_pair is None else current_pair
            q_drop = q
            q_drop_values[j] = q_drop
            Pr_Sup_p[j] = float(g1) * phix_bsa
            Pr_Drop_p[j] = float(g2) * phix_bsa

            state = reservoir(i)
            alpha1, g_sup, h_sup = state.partition(g1)
            alpha2, g_drop, h_drop = state.partition(g2, q=q_drop)

            Pol_Sup_p[j] = alpha1 * i * phix_peg
            Pol_Drop_p[j] = alpha2 * i * phix_peg
            a_Sup_p[j] = alpha1
            a_Drop_p[j] = alpha2
            phi_R[j] = i

            qx_Sup_p[j], dPidPhiP_Sup_p[j] = state.endpoint_qx, state.endpoint_dpi
            qx_Drop_p[j], dPidPhiP_Drop_p[j] = eos.polymer.qxf(q_drop, i)

            mu_drop, pi_drop = state.evaluate(g2, q=q_drop)
            mu_sup, pi_sup = state.evaluate(g1)
            mu0_drop, p0_drop = eos.reference(g2)
            mu0_sup, p0_sup = eos.reference(g1)
            pressure_scale = phix_bsa * Rb * T

            Pi_Drop[j] = pi_drop * pressure_scale
            Pi_Sup[j] = pi_sup * pressure_scale
            Mu_Drop[j] = mu_drop
            Mu_Sup[j] = mu_sup
            P0_Drop[j] = p0_drop * pressure_scale
            P0_Sup[j] = p0_sup * pressure_scale
            Pp_Drop[j] = (pi_drop - p0_drop) * pressure_scale
            Pp_Sup[j] = (pi_sup - p0_sup) * pressure_scale
            Mu0_Drop[j] = mu0_drop
            Mu0_Sup[j] = mu0_sup
            MuR_Drop[j] = mu_drop - mu0_drop
            MuR_Sup[j] = mu_sup - mu0_sup
            g_Drop[j] = g_drop
            g_Sup[j] = g_sup
            h_Drop[j] = h_drop
            h_Sup[j] = h_sup

        params = [
            Rpr_in,
            Rpol_in,
            mbsa,
            mpeg,
            solvent,
            N_steps,
            PhiP_min,
            PhiP_max,
            outguess_c,
            outguess_tp,
            guess_b,
            TC,
            phix_bsa,
            phix_peg,
        ]

        self.vbsa = vbsa
        self.vpeg = vpeg
        self.q = q
        self.phix_bsa = phix_bsa
        self.phix_peg = phix_peg
        self.PhiP_min = PhiP_min
        self.PhiP_max = PhiP_max
        self.outguess_c = outguess_c
        self.outguess_tp = outguess_tp
        if endpoint is not None:
            self.critical_endpoint = endpoint
        self.s = 0
        self.q_Drop_p = q_drop_values
        self.q_Sup_p = np.full(N_steps, q)

        self._assign_return_values(
            [
                phi_R,
                Pr_Drop_p,
                Pol_Drop_p,
                a_Drop_p,
                Pr_Sup_p,
                Pol_Sup_p,
                a_Sup_p,
                qx_Drop_p,
                dPidPhiP_Drop_p,
                qx_Sup_p,
                dPidPhiP_Sup_p,
                Pi_Drop,
                P0_Drop,
                Pp_Drop,
                Mu_Drop,
                Mu0_Drop,
                MuR_Drop,
                Pi_Sup,
                P0_Sup,
                Pp_Sup,
                Mu_Sup,
                Mu0_Sup,
                MuR_Sup,
                cp,
                cp_list,
                tp,
                tp_list,
                g_Drop,
                g_Sup,
                h_Drop,
                h_Sup,
                params,
            ]
        )
        return self

    def GFVT_full(
        self,
        PhiP_max: float = 1,
        *,
        critical: bool | None = None,
        progress: bool | None = None,
        verbose: bool | None = None,
    ) -> "PD":
        """Continue the fluid binodal to a specified reservoir concentration.

        PhiP_max is dimensionless phi_R and must be >= the critical value.
        Continuation starts near criticality even when the requested limit lies
        beyond the triple point. Results retain the legacy descending order.
        """
        return self.GFVT(
            PhiP_max_override=PhiP_max,
            critical=critical,
            progress=progress,
            verbose=verbose,
        )

    def GFVT_es(
        self,
        PhiP_max: float = 1,
        *,
        Z: float | None = None,
        ionic_strength: float | None = None,
        m: float | None = None,
        bjerrum_length: float = 0.71e-9,
        depletion_exponent: float | None = None,
        critical: bool | None = None,
        progress: bool | None = None,
        verbose: bool | None = None,
    ) -> "PD":
        """Continue the paper's electrostatic fluid binodal to PhiP_max.

        Martens et al., doi:10.1016/j.jcis.2026.141573, Eqs. (1)--(13).
        Supply either m >= 1 directly, or signed effective charge Z and TOTAL
        ionic_strength in mol/L (salt plus buffer). Z=0 gives m=1 and does not
        require ionic_strength. bjerrum_length is in metres (default: room-
        temperature water); supply the appropriate value at other temperatures.

        Uses solvent='good'. The depletion exponent defaults to the paper's
        1.53 when nu=None; m=1, depletion_exponent=1.54 then recovers
        GFVT_full fluid results. With explicit nu, both models instead share
        exponents derived from nu, including depletion_exponent=2*gamma.
        An explicitly supplied, conflicting depletion exponent is rejected.
        PhiP_max is dimensionless reservoir phi_R, at least its critical value.
        Starts just above criticality, follows distinct fluids in log density,
        and stores arrays in the same descending order as GFVT_full. Physical
        concentrations use Rpr_in, not the electrostatic effective radius.

        critical=True also computes the fluid critical locus on the existing
        q=10...0.388 grid, at fixed m. No crystal EOS is supplied by this paper:
        tp/outguess_tp are None and tp_list is empty. es_m, es_Z,
        es_ionic_strength, es_bjerrum_length, es_kappa, es_d_eff and
        es_depletion_exponent retain the model parameters. Pickle the PD object
        to preserve them; the legacy 32-item list contains only numeric results.
        The paper's formal perturbation range is m <= approximately 1.5.
        """
        from .electrostatics import calculate_es

        return calculate_es(
            self, PhiP_max, Z=Z, ionic_strength=ionic_strength, m=m,
            bjerrum_length=bjerrum_length, depletion_exponent=depletion_exponent,
            critical=critical, progress=progress, verbose=verbose,
        )

    def GFVT_rods(
        self,
        PhiP_max: float = 1,
        *,
        L_rod: float | None = None,
        L: float | None = None,
        coexistence: str = "II",
        dep_model: str = "int",
        odf: str = "gaussian",
        progress: bool | None = None,
        verbose: bool | None = None,
        n_orient: int = 128,
        n_azimuth: int = 256,
        progress_callback=None,
    ) -> "PD":
        """Calculate spherocylinder coexistence using the 2011 book theory.

        L_rod is the cylindrical length excluding the hemispherical end caps,
        in metres. D=2*Rpr_in and sigma=2*Rpol_in are inherited from PD.
        L is an alias for L_rod. Select coexistence='II', 'IN', 'NN', or 'all';
        dep_model='int' uses interacting polymers and 'phs' penetrable hard
        spheres. The interacting closure requires solvent='good' and nu=None.
        q stores the nominal sigma/D; the qx_* diagnostics store the effective
        insertion size 2*delta/D, which varies with reservoir concentration
        for interacting polymers. odf='gaussian' uses the Gaussian approximation;
        'numerical' minimizes the orientational distribution numerically.

        progress_callback, when supplied, receives dictionaries with stage,
        completed, total, and (during tracing) phi_R. Totals include refined
        reservoir samples. Callback errors warn without interrupting physics.

        rod_branches stores every continuous coexistence branch, including its
        phase identities and stability mask. Standard legacy arrays are also
        available when the result is one continuous branch. Save the whole PD
        object to retain the rod theory and branch metadata.

        The previous isotropic-only calculation with spherical polymer
        depletion thickness is available explicitly as GFVT_rods_legacy().
        """
        from .rod_phase import calculate_rods
        self._reject_kuhn_variant()
        return calculate_rods(
            self, PhiP_max, L_rod=L_rod, L=L, coexistence=coexistence,
            dep_model=dep_model, odf=odf, progress=progress, verbose=verbose,
            n_orient=n_orient, n_azimuth=n_azimuth,
            progress_callback=progress_callback,
        )

    def GFVT_rods_legacy(
        self,
        PhiP_max: float = 1,
        *,
        D: float | None = None,
        L: float = 0,
        progress: bool | None = None,
        verbose: bool | None = None,
    ) -> "PD":
        """Retain the previous isotropic-only spherocylinder approximation.

        D (default 2*Rpr_in) and L (default zero) are in metres; L is the
        cylindrical length, excluding end caps. Both phases have the same
        geometry. The rod critical point is recalculated, then adaptive
        continuation follows distinct fluids up to dimensionless PhiP_max.

        Uses isotropic SPT reference/free-volume expressions from Lekkerkerker,
        Tuinier & Vis (2024), Chapter 8, Eqs. 8.38--8.54. The L=0 limit is SPT
        spheres, not the Carnahan--Starling reference used by GFVT_full.
        For good/theta/RG polymers, the existing spherical depletion-thickness
        fit at radius D/2 is retained as an approximation; reservoir pressure
        and all concentration conversions use the actual rod volume.

        molplot works directly in mM or g/L. No nematic/solid stability or
        triple points are calculated (tp/outguess_tp=None; locus lists empty).
        self.critical does not request a locus scan here: the single critical
        point is always solved. D/L metadata survive pickling the PD object,
        but are not encoded in its legacy 32-item list.
        """
        from .rods import calculate_rods
        self._reject_kuhn_variant()
        result = calculate_rods(self, PhiP_max, D=D, L=L, progress=progress, verbose=verbose)
        self.rod_theory = "legacy2024_sphere_depletion"
        return result

    def GFVT_irods_scale(
        self,
        s: float = 0,
        PhiP_max: float = 1,
        *,
        D: float | None = None,
        progress: bool | None = None,
        verbose: bool | None = None,
    ) -> "PD":
        """Solve isotropic coexistence with density-dependent droplet rod length.

        D defaults to 2*Rpr_in and is measured in metres. Supernatant proteins
        have L=0, while droplet proteins have L=D*s*(phi_drop-phi_c), where
        phi_drop is the actual occupied volume fraction, s is dimensionless
        and non-negative, and phi_c is the fixed L=0 reference critical value.
        The spherocylinder volume includes its hemispherical end caps.

        This constrained, phenomenological shape law is differentiated in the
        free energy when calculating chemical potential and pressure. No extra
        conformational free energy is included. A nonzero s need not preserve
        the reference critical point; it is not presented as a computed critical
        point of the modified model. Nematic and solid phases are excluded.

        Coexistence is traced from the reference onset to PhiP_max using
        supporting common tangents. For s>0 the free energy has a corner at
        phi_c: boundary_Drop_p identifies states with phi_drop=phi_c and L=0.
        corner_fraction_p selects a slope between the one-sided derivatives;
        it is NOT a phase fraction. These boundary states can persist even
        for positive s. cp=None avoids drawing a false smooth critical point;
        onset/reference_outguess_c retain the reference coordinates.
        Require 1-1.5*s*phi_c>0 for a monotonic number-density mapping.

        Stored Pr_Drop_p/Pr_Sup_p already use each phase's actual molar volume.
        molplot or molplot_scale(..., s=self.s) plots those concentrations without
        applying the volume correction twice. Per-phase Phi_*, L_*, vbsa_* and
        phix_bsa_* arrays retain the geometry and conversion at every point.
        s=0 reproduces the L=0 GFVT_rods_legacy model.
        """
        from .irods import calculate_irods_scale
        self._reject_kuhn_variant()
        return calculate_irods_scale(
            self, s=s, PhiP_max=PhiP_max, D=D, progress=progress, verbose=verbose,
        )

    def GFVT_scale(
        self,
        PhiP_max: float = 1,
        *,
        s: float = 0,
        nu: float | None = None,
        width: float = 0.1,
        critical: bool | None = None,
        progress: bool | None = None,
        verbose: bool | None = None,
    ) -> "PD":
        """GFVT with one consistent, optionally density-dependent exponent.

        A good-solvent calculation uses nu=0.588 unless nu is explicitly
        supplied here or on PD. For theta solvent the default is nu=0.5.
        gamma=nu/(3*nu-1) sets both the depletion concentration exponent
        (2*gamma) and osmotic-pressure exponent (3*gamma). The pressure
        derivative is taken from that same pressure equation.

        The optional droplet-side law is
        ``nu_eff(phi)=nu*(1+s*H_width(phi-phi_c))``, where H is zero on the
        dilute side and ``x*exp(-width/x)`` for positive x. The supernatant and
        crystal therefore retain the base nu. H and all derivatives vanish at
        phi_c, avoiding the kink produced by ``max(phi-phi_c, 0)``. All density chain
        rules from nu_eff are included in the semigrand free energy, chemical
        potential and pressure. This is a phenomenological density coupling,
        not an ordinary polymer reservoir equation of state. Only good-solvent
        calculations support s>0; width must be positive.

        The ordinary spherical geometry and dilute Rg are retained. For s=0,
        this is exactly GFVT_full with the selected constant nu. For s>0,
        critical/triple points and coexistence are recalculated using the
        modified free energy, continuing from near criticality to PhiP_max.
        critical=True additionally recomputes their loci over size ratio q,
        including the critical endpoint that sets the lower scan limit.
        critical_endpoint then records q, phi_R, phi_fluid and phi_solid.
        GFVT_exclusion_scale remains the separate depletion-length model.
        """
        from copy import copy
        from .polymer import GOOD_SOLVENT_NU
        self._reject_kuhn_variant()

        candidate = copy(self)
        if nu is None:
            nu = getattr(self, "nu", None)
        if nu is None:
            nu = 0.5 if self.solvent == "theta" else GOOD_SOLVENT_NU
        PolymerScaling(self.solvent, nu)
        candidate.nu = nu
        from .nu_scale import calculate_nu_scale
        calculate_nu_scale(candidate, s, PhiP_max, nu=nu, width=width,
                           critical=critical, progress=progress, verbose=verbose)
        # Replace only after a successful solve, including removal of stale
        # density-dependent or electrostatic metadata from previous results.
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)
        return self

    def GFVT_exclusion_scale(
        self,
        s: float = 0,
        PhiP_max: float = 1,
        *,
        width: float = 0.1,
        critical: bool | None = None,
        progress: bool | None = None,
        verbose: bool | None = None,
    ) -> "PD":
        """Separate, optional density-dependent exclusion-layer model.

        delta_eff = delta_GFVT + Rpr_in*s*H_width(phi-phi_c), where phi_c is
        the ordinary GFVT critical fraction. For x>0,
        H_w(x) = x*exp(-w/x); for x<=0 it is zero. This elementary ramp and
        its derivatives require no quadrature or special functions.
        width (default 0.1) is in protein-volume-fraction units; s>=0 sets the
        limiting slope. The ramp and all derivatives vanish at the critical
        density, while 0<=H_w'(x)<=1. Free energy, chemical potential, pressure
        and their critical derivatives are continuous. Their full density
        chain rules are included throughout the polymer reservoir integral.

        Protein size, physical q, concentration conversions and reservoir
        osmotic EOS remain fixed. s=0 exactly reproduces GFVT_full. The smooth
        critical point equals the ordinary one, and is stored as cp/outguess_c.
        Continuation starts near criticality and checks supporting common
        tangents. An additional first-order transition, if present for the
        chosen parameters, is preserved and marked in coexistence_breaks_p.

        width=0 explicitly selects the former sharp max(phi-phi_c,0) law,
        whose threshold has one-sided derivatives and cp=None for s>0.
        exclusion_width/exclusion_law record the choice ("smooth_exponential"
        for positive width). Recalculate results from the former erfc-based
        ramp to use this law; equal widths do not give equal binodals.
        For positive s no
        modified crystal EOS is assumed: tp=None; reference_* retains ordinary
        landmarks. critical=True scans q with the ramp anchored at each q's
        ordinary critical density; smooth cp_list contains those criticals.

        molplot and molplot_scale(..., s=self.s) use physical concentrations
        without further volume rescaling. Pickle the full PD object to retain
        the exclusion-law metadata; old saved results keep their old meaning.
        """
        from .exclusion_scale import calculate_exclusion_scale
        self._reject_kuhn_variant()
        return calculate_exclusion_scale(
            self, s=s, PhiP_max=PhiP_max, width=width, critical=critical,
            progress=progress, verbose=verbose,
        )

    @staticmethod
    def molplot_dat_err_csv(
        data,
        plabel,
        pcolor,
        mbsa=0,
        mpeg=0,
        n_tie=False,
        plot_tot=False,
        ax=None,
        show_legend=False,
    ):
        """Plot experimental binodal CSV data with error bars."""
        ax = PD._get_axis(ax)
        (
            BN_BSA_tot,
            BN_BSA_tot_err,
            BN_BSA_sup,
            BN_BSA_sup_err,
            BN_BSA_drop,
            BN_BSA_drop_err,
            BN_PEG_tot,
            BN_PEG_tot_err,
            BN_PEG_sup,
            BN_PEG_sup_err,
            BN_PEG_drop,
            BN_PEG_drop_err,
            CP_BSA,
            CP_BSA_err,
            CP_PEG,
            CP_PEG_err,
        ) = PD._parse_binodal_data(data, mbsa=mbsa, mpeg=mpeg)

        PD._plot_data_errorbars(
            ax,
            plabel,
            pcolor,
            BN_BSA_tot,
            BN_BSA_tot_err,
            BN_BSA_sup,
            BN_BSA_sup_err,
            BN_BSA_drop,
            BN_BSA_drop_err,
            BN_PEG_tot,
            BN_PEG_tot_err,
            BN_PEG_sup,
            BN_PEG_sup_err,
            BN_PEG_drop,
            BN_PEG_drop_err,
            CP_BSA,
            CP_BSA_err,
            CP_PEG,
            CP_PEG_err,
            n_tie=n_tie,
            plot_tot=plot_tot,
        )
        PD._maybe_legend(ax, show_legend)
        return ax

    def molplot_dat_err_csv_phi(
        self,
        data,
        plabel,
        pcolor,
        ax=None,
        mbsa=0,
        mpeg=0,
        n_tie=False,
        plot_tot=False,
        gamma=None,
        show_legend=False,
    ):
        """Plot experimental CSV data in volume-fraction coordinates."""
        self._require_results()
        ax = self._get_axis(ax)
        q = self.Rpol_in / self.Rpr_in
        gamma = self._axis_gamma(gamma)
        qq = 1 if gamma is None else q ** -(1 / gamma)
        mbsa = self.mbsa if mbsa == 0 else mbsa
        mpeg = self.mpeg if mpeg == 0 else mpeg

        (
            BN_BSA_tot,
            BN_BSA_tot_err,
            BN_BSA_sup,
            BN_BSA_sup_err,
            BN_BSA_drop,
            BN_BSA_drop_err,
            BN_PEG_tot,
            BN_PEG_tot_err,
            BN_PEG_sup,
            BN_PEG_sup_err,
            BN_PEG_drop,
            BN_PEG_drop_err,
            CP_BSA,
            CP_BSA_err,
            CP_PEG,
            CP_PEG_err,
        ) = self._parse_binodal_data(
            data,
            mbsa=mbsa,
            mpeg=mpeg,
            phix_bsa=self.phix_bsa,
            phix_peg=self.phix_peg,
            peg_scale=qq,
        )

        self._plot_data_errorbars(
            ax,
            plabel,
            pcolor,
            BN_BSA_tot,
            BN_BSA_tot_err,
            BN_BSA_sup,
            BN_BSA_sup_err,
            BN_BSA_drop,
            BN_BSA_drop_err,
            BN_PEG_tot,
            BN_PEG_tot_err,
            BN_PEG_sup,
            BN_PEG_sup_err,
            BN_PEG_drop,
            BN_PEG_drop_err,
            CP_BSA,
            CP_BSA_err,
            CP_PEG,
            CP_PEG_err,
            n_tie=n_tie,
            plot_tot=plot_tot,
        )
        self._maybe_legend(ax, show_legend)
        return ax

    def molplot(
        self,
        plabel,
        pcolor,
        crit_color=False,
        FVT_color=False,
        n_tie=0,
        mbsa=0,
        mpeg=0,
        ax=None,
        show_legend=False,
        mass=False,
        plim=500,
        *,
        branch=None,
        show_metastable=False,
    ):
        """Plot the GFVT binodal in concentration or mass coordinates."""
        return self.molplot_scale(
            plabel, pcolor, crit_color=crit_color, FVT_color=FVT_color,
            n_tie=n_tie, mbsa=mbsa, mpeg=mpeg, ax=ax,
            show_legend=show_legend, mass=mass, plim=plim,
            branch=branch, show_metastable=show_metastable,
        )

    def molplot_scale(
        self,
        plabel,
        pcolor,
        crit_color=False,
        FVT_color=False,
        n_tie=0,
        mbsa=0,
        mpeg=0,
        ax=None,
        show_legend=False,
        mass=False,
        plim=500,
        *,
        s=0,
        branch=None,
        show_metastable=False,
    ):
        """Plot a binodal using geometry-aware protein concentration conversions.

        For spherical results, the droplet radius is Rpr_in*f, where
        f=1+s*(phi_drop-phi_c); the droplet concentration is Pr_Drop_p/f**3.
        For fixed GFVT_rods_legacy results, keep D and supernatant L fixed and use
        L_drop=L+D*s*(phi_drop-phi_c) solely for this plotting conversion.
        Its droplet molar volume is Nav*(pi*D**2*L_drop/4+pi*D**3/6).

        GFVT_scale changes the polymer exponent. Its protein concentrations
        are already physical and receive no geometry rescaling; leave s=0 or
        pass the value used in the calculation.
        GFVT_exclusion_scale results change only the PEG exclusion layer. Their
        protein concentrations are already physical and receive no rescaling.
        Leave s=0 or use the solved s; a different nonzero s requires a new solve.
        Older untagged saved spherical results retain the conversion above.

        GFVT_irods_scale results already store phase-specific molar volumes and
        physical concentrations. These volumes are used directly, without a
        second scaling. Leave s=0 (the default), or pass the value used in that
        solve; a different nonzero s requires rerunning GFVT_irods_scale.

        s=0 reproduces molplot. Supernatant and PEG coordinates stay unchanged;
        droplet corrections also apply to highlighted segments and tie lines.
        Stored results are never modified. Critical markers are drawn only when
        a calculated critical point is available (cp is not None). For book-model
        GFVT_rods results, s must be zero; branch selects II/IN/NN/all and
        show_metastable adds dashed continuations.
        """
        if getattr(self, "geometry", "spheres") == "spherocylinders":
            self._reject_fvt_overlay(FVT_color)
            if not np.isscalar(s) or not np.isreal(s) or not np.isfinite(s):
                raise ValueError("s must be a finite real scalar.")
            if s != 0:
                raise ValueError("GFVT_rods has fixed rod dimensions; molplot_scale requires s=0.")
            from .rod_plot import plot_rod_branches
            return plot_rod_branches(
                self, plabel, pcolor, coordinates="concentration", ax=ax,
                mass=mass, mbsa=mbsa, mpeg=mpeg, branch=branch,
                show_metastable=show_metastable, n_tie=n_tie,
                crit_color=crit_color, show_legend=show_legend,
            )
        self._require_results()
        self._reject_fvt_overlay(FVT_color)
        if not np.isscalar(s) or not np.isreal(s) or not np.isfinite(s):
            raise ValueError("s must be a finite real scalar.")

        geometry = getattr(self, "geometry", "spheres")
        bsa_drop, bsa_sup = self.Pr_Drop_p, self.Pr_Sup_p
        if getattr(self, "model_kind", None) == "polymer_scaling":
            if s != 0:
                raise ValueError("GFVT_scale has no density-dependent scaling; use molplot or s=0.")
        elif getattr(self, "model_kind", None) == "density_scaled_nu":
            if s != 0 and not np.isclose(s, self.s, rtol=1e-12, atol=0):
                raise ValueError(
                    "s must match the GFVT_scale calculation; rerun GFVT_scale to use a different s."
                )
        elif getattr(self, "model_kind", None) == "scaled_exclusion_layer":
            if s != 0 and not np.isclose(s, self.s, rtol=1e-12, atol=0):
                raise ValueError(
                    "s must match the GFVT_exclusion_scale calculation; rerun GFVT_exclusion_scale "
                    "to use a different nonzero s."
                )
        elif geometry == "isotropic_rods_scaled":
            if s != 0 and not np.isclose(s, self.s, rtol=1e-12, atol=0):
                raise ValueError(
                    "s must match the GFVT_irods_scale calculation; rerun "
                    "GFVT_irods_scale to use a different nonzero s."
                )
            bsa_drop = np.asarray(self.Phi_Drop_p) / np.asarray(self.vbsa_Drop_p)
            bsa_sup = np.asarray(self.Phi_Sup_p) / np.asarray(self.vbsa_Sup_p)
        elif s != 0 and geometry == "isotropic_rods":
            phix_bsa, _ = self._get_phix_scales()
            phi_drop = np.asarray(self.Pr_Drop_p, dtype=float) / phix_bsa
            phi_c = self.cp[0] / phix_bsa
            length_drop = self.L + self.D * s * (phi_drop - phi_c)
            if np.any(~np.isfinite(length_drop) | (length_drop < 0)):
                raise ValueError("s gives a negative or non-finite droplet spherocylinder length.")
            volume_drop = 6.022e23 * (
                np.pi * self.D**2 * length_drop / 4 + np.pi * self.D**3 / 6
            )
            bsa_drop = phi_drop / volume_drop
        elif s != 0:
            phix_bsa, _ = self._get_phix_scales()
            phi_drop = np.asarray(self.Pr_Drop_p, dtype=float) / phix_bsa
            phi_c = self.cp[0] / phix_bsa
            radius_factor = 1 + s * (phi_drop - phi_c)
            if np.any(~np.isfinite(radius_factor) | (radius_factor <= 0)):
                raise ValueError(
                    "s gives a non-positive or non-finite droplet radius: "
                    "1 + s*(phi - phi_c) must be positive and finite at every point."
                )
            bsa_drop = np.asarray(self.Pr_Drop_p, dtype=float) / radius_factor**3

        ax = self._get_axis(ax)
        mbsa, mpeg = self._plot_masses(mbsa, mpeg)
        full_label = self._full_label(plabel)

        if mass:
            b_bsa_A = bsa_drop * mbsa
            b_bsa_B = bsa_sup * mbsa
            b_PEG_A = self.Pol_Drop_p * mpeg
            b_PEG_B = self.Pol_Sup_p * mpeg
        else:
            b_bsa_A = bsa_drop
            b_bsa_B = bsa_sup
            b_PEG_A = self.Pol_Drop_p
            b_PEG_B = self.Pol_Sup_p

        idx = np.abs(np.asarray(self.Pol_Sup_p) - plim).argmin()
        self._plot_binodal_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, full_label)
        highlight_x, highlight_y = self._drop_line_coordinates(b_bsa_A[idx:], b_PEG_A[idx:], start=idx)
        ax.plot(highlight_x, highlight_y, "-", color=pcolor, linewidth=10, alpha=0.4)
        ax.plot(b_bsa_B[idx:], b_PEG_B[idx:], "-", color=pcolor, linewidth=10, alpha=0.4)
        self._plot_tie_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, n_tie)
        self._plot_critical_lines(ax, crit_color, full_label, mass=mass, mbsa=mbsa, mpeg=mpeg)
        if self.cp is not None:
            c_bsa = self.cp[0] * (mbsa if mass else 1)
            c_PEG = self.cp[1] * (mpeg if mass else 1)
            ax.scatter(c_bsa, c_PEG, s=300, color=pcolor, marker="*", label=full_label + " critical point")
        self._maybe_legend(ax, show_legend)
        return ax

    def molplot_phi(
        self,
        plabel,
        pcolor,
        ax=None,
        crit_color=False,
        FVT_color=False,
        n_tie=0,
        gamma=None,
        show_legend=False,
        *,
        branch=None,
        show_metastable=False,
    ):
        """Plot the GFVT binodal in volume-fraction coordinates."""
        if getattr(self, "geometry", "spheres") == "spherocylinders":
            self._reject_fvt_overlay(FVT_color)
            from .rod_plot import plot_rod_branches
            return plot_rod_branches(
                self, plabel, pcolor, coordinates="phi", ax=ax, gamma=gamma,
                branch=branch, show_metastable=show_metastable,
                n_tie=n_tie, crit_color=crit_color, show_legend=show_legend,
            )
        self._require_results()
        self._reject_fvt_overlay(FVT_color)
        ax = self._get_axis(ax)
        full_label = self._full_label(plabel)
        q = self.q
        phix_bsa, phix_peg = self._get_phix_scales()
        
        gamma = self._axis_gamma(gamma)
        qq = 1 if gamma is None else q ** -(1 / gamma)

        b_bsa_Drop, b_bsa_Sup = self._protein_volume_fractions()
        b_PEG_Drop = qq * self.Pol_Drop_p / phix_peg
        b_PEG_Sup = qq * self.Pol_Sup_p / phix_peg

        b_PEG_A = np.asarray([R*A*qq for R, A in zip(self.phi_R, self.a_Drop_p)]) 
        b_PEG_B = np.asarray([R*A*qq for R, A in zip(self.phi_R, self.a_Sup_p)]) 

        drop_x, drop_y = self._drop_line_coordinates(b_bsa_Drop, b_PEG_A)
        ax.plot(drop_x, drop_y, "--", color='black', linewidth=2)
        ax.plot(b_bsa_Sup, b_PEG_B, "--", color='black', linewidth=2)

        self._plot_binodal_lines(ax, b_bsa_Drop, b_PEG_Drop, b_bsa_Sup, b_PEG_Sup, pcolor, full_label)
        self._plot_tie_lines(ax, b_bsa_Drop, b_PEG_Drop, b_bsa_Sup, b_PEG_Sup, pcolor, n_tie)
        self._plot_phi_critical_lines(ax, crit_color, full_label, gamma=gamma)
        if self.cp is not None:
            c_bsa = self.cp[0] / phix_bsa
            c_PEG = qq * self.cp[1] / phix_peg
            ax.scatter(c_bsa, c_PEG, s=300, color=pcolor, marker="*", label=full_label + " critical")
        self._maybe_legend(ax, show_legend)
        return ax

    def molplot_phiR(
        self,
        plabel,
        pcolor,
        crit_color=False,
        FVT_color=False,
        n_tie=0,
        ax=None,
        show_legend=False,
        *,
        branch=None,
        show_metastable=False,
    ):
        """Plot the GFVT binodal against reservoir polymer volume fraction."""
        if getattr(self, "geometry", "spheres") == "spherocylinders":
            self._reject_fvt_overlay(FVT_color)
            from .rod_plot import plot_rod_branches
            return plot_rod_branches(
                self, plabel, pcolor, coordinates="phiR", ax=ax,
                branch=branch, show_metastable=show_metastable,
                n_tie=n_tie, crit_color=crit_color, show_legend=show_legend,
            )
        self._require_results()
        self._reject_fvt_overlay(FVT_color)
        ax = self._get_axis(ax)
        full_label = self._full_label(plabel)
        q = self.Rpol_in / self.Rpr_in
        phix_bsa, phix_peg = self._get_phix_scales()
        qq = 1
        color = crit_color or pcolor

        b_bsa_A, b_bsa_B = self._protein_volume_fractions()
        b_PEG_A = np.asarray(self.phi_R) * qq
        b_PEG_B = np.asarray(self.phi_R) * qq

        self._plot_binodal_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, full_label)
        self._plot_tie_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, n_tie)
        if self.cp is not None:
            ax.scatter(self.outguess_c[0], qq * self.outguess_c[1], color=color, s=300, label=full_label + " critical point")
        if getattr(self, "geometry", "spheres") == "spheres" and self.outguess_tp is not None:
            gamma, _, _, y_a_L, bsa_a_L, bsa_a_G = self._analytical_base_curves()
            ax.plot(bsa_a_L, y_a_L * qq, "--", color=color, label=full_label + " analytical Liquid")
            ax.plot(bsa_a_G, y_a_L * qq, "--", color=color, label=full_label + " analytical Gas")
        self._maybe_legend(ax, show_legend)
        return ax

    def molplot_phi_analytical(
        self,
        plabel,
        pcolor,
        ax=None,
        crit_color=False,
        FVT_color=False,
        n_tie=0,
        show_legend=False,
    ):
        """Plot GFVT volume-fraction binodal with analytical approximations."""
        self._require_results()
        self._reject_fvt_overlay(FVT_color)
        ax = self._get_axis(ax)
        full_label = self._full_label(plabel)
        color = crit_color or pcolor
        phix_bsa, phix_peg = self._get_phix_scales()
        _, _, _, y_a_L, bsa_a_L, bsa_a_G = self._analytical_base_curves()

        b_bsa_A = self.Pr_Drop_p / phix_bsa
        b_bsa_B = self.Pr_Sup_p / phix_bsa
        b_PEG_A = self.Pol_Drop_p / phix_peg
        b_PEG_B = self.Pol_Sup_p / phix_peg
        phi_a_L = [self._alpha(Phi, Phi_pol) * Phi_pol for Phi_pol, Phi in zip(y_a_L, bsa_a_L)]
        phi_a_G = [self._alpha(Phi, Phi_pol) * Phi_pol for Phi_pol, Phi in zip(y_a_L, bsa_a_G)]

        self._plot_binodal_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, full_label)
        self._plot_tie_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, n_tie)
        ax.plot(bsa_a_L, phi_a_L, "--", color=color, label=full_label + " analytical Liquid")
        ax.plot(bsa_a_G, phi_a_G, "--", color=color, label=full_label + " analytical Gas")
        self._maybe_legend(ax, show_legend)
        return ax

    def molplot_analytical(
        self,
        plabel,
        pcolor,
        ax=None,
        crit_color=False,
        mbsa=0,
        mpeg=0,
        FVT_color=False,
        n_tie=0,
        show_legend=False,
        mass=False,
    ):
        """Plot GFVT concentration binodal with analytical approximations."""
        self._require_results()
        self._reject_fvt_overlay(FVT_color)
        ax = self._get_axis(ax)
        mbsa, mpeg = self._plot_masses(mbsa, mpeg)
        full_label = self._full_label(plabel)
        color = crit_color or pcolor
        _, _, _, y_a_L, phi_bsa_a_L, phi_bsa_a_G = self._analytical_base_curves()

        if mass:
            b_bsa_A = self.Pr_Drop_p * mbsa
            b_bsa_B = self.Pr_Sup_p * mbsa
            b_PEG_A = self.Pol_Drop_p * mpeg
            b_PEG_B = self.Pol_Sup_p * mpeg
            phi_a_L = [
                self._alpha(Phi, Phi_pol) * Phi_pol * self.phix_peg * mpeg
                for Phi_pol, Phi in zip(y_a_L, phi_bsa_a_L)
            ]
            phi_a_G = [
                self._alpha(Phi, Phi_pol) * Phi_pol * self.phix_peg * mpeg
                for Phi_pol, Phi in zip(y_a_L, phi_bsa_a_G)
            ]
            bsa_a_L = [Phi * self.phix_bsa * mbsa for Phi in phi_bsa_a_L]
            bsa_a_G = [Phi * self.phix_bsa * mbsa for Phi in phi_bsa_a_G]
        else:
            b_bsa_A = self.Pr_Drop_p
            b_bsa_B = self.Pr_Sup_p
            b_PEG_A = self.Pol_Drop_p
            b_PEG_B = self.Pol_Sup_p
            phi_a_L = [
                self._alpha(Phi, Phi_pol) * Phi_pol * self.phix_peg
                for Phi_pol, Phi in zip(y_a_L, phi_bsa_a_L)
            ]
            phi_a_G = [
                self._alpha(Phi, Phi_pol) * Phi_pol * self.phix_peg
                for Phi_pol, Phi in zip(y_a_L, phi_bsa_a_G)
            ]
            bsa_a_L = [Phi * self.phix_bsa for Phi in phi_bsa_a_L]
            bsa_a_G = [Phi * self.phix_bsa for Phi in phi_bsa_a_G]

        self._plot_binodal_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, full_label)
        self._plot_tie_lines(ax, b_bsa_A, b_PEG_A, b_bsa_B, b_PEG_B, pcolor, n_tie)
        ax.plot(bsa_a_L, phi_a_L, "--", color=color, label=full_label + " analytical Liquid")
        ax.plot(bsa_a_G, phi_a_G, "--", color=color, label=full_label + " analytical Gas")
        self._maybe_legend(ax, show_legend)
        return ax

    def d_Rg(
        self,
        pcolor,
        plabel,
        ax=None,
        plim=100,
        *,
        mass=False,
        highlight=True,
        n_points=100000,
    ):
        """Plot the polymer depletion thickness approximation."""
        self._require_results()
        ax = self._get_axis(ax)
        q = self.Rpol_in / self.Rpr_in
        x_scale = self.phix_peg * (self.mpeg if mass else 1)
        phi_r = np.asarray(self.phi_R)
        if getattr(self, "model_kind", None) == "density_scaled_nu":
            ax.plot(phi_r * x_scale, self.Rpr_in * 1e9 * np.asarray(self.qx_Sup_p),
                    label=plabel + " supernatant", color=pcolor)
            ax.plot(phi_r * x_scale, self.Rpr_in * 1e9 * np.asarray(self.qx_Drop_p),
                    label=plabel + " droplet", color=pcolor, linestyle="--")
            ax.scatter(phi_r[-1] * x_scale, self.Rpr_in * 1e9 * self.qx_Drop_p[-1],
                       color=pcolor, s=150)
            return ax
        phi_grid = np.linspace(0, 2000, n_points)
        phi_grid_approx = np.linspace(0, 2000, n_points)

        polymer = self._polymer_scaling()
        exponent = getattr(self, "es_depletion_exponent", None)
        depletion = self.Rpr_in * 1e9 * polymer.depletion(q, phi_grid, exponent=exponent)
        ax.plot(phi_grid * x_scale, depletion, label=plabel, color=pcolor)
        if getattr(self, "kuhn_crossover", None) is None:
            depletion_approx = self.Rpr_in * 1e9 * polymer.depletion(
                q, phi_grid_approx, exponent=exponent, semidilute=True,
            )
            ax.plot(phi_grid_approx * x_scale, depletion_approx, color=pcolor, linestyle="--", linewidth=1)
        ax.scatter(phi_r[-1] * x_scale, self.Rpr_in * 1e9 * self.qx_Drop_p[-1], color=pcolor, s=150)

        idx = np.abs(np.asarray(self.Pol_Sup_p) - plim).argmin()
        start = idx if highlight else 0
        linewidth = 10 if highlight else 5
        ax.plot(
            phi_r[start:] * x_scale,
            self.Rpr_in * 1e9 * np.asarray(self.qx_Sup_p)[start:],
            color=pcolor,
            linewidth=linewidth,
            alpha=0.6,
        )
        return ax

    def pi_scale(
        self,
        pcolor,
        plabel,
        ax=None,
        plim=100,
        *,
        mass=False,
        reduced=False,
        gamma=None,
        highlight=True,
        secondary_axis=None,
        n_points=100000,
    ):
        """Plot the reservoir pressure used in the calculation.

        gamma, if supplied, must match the model; set nu on PD to change it.
        """
        self._require_results()
        ax = self._get_axis(ax)
        q = self.Rpol_in / self.Rpr_in
        x_scale = self.phix_peg * (self.mpeg if mass else 1)
        y_scale = q**-3 if (mass or reduced) else self.phix_peg
        phi_r = np.asarray(self.phi_R)
        phi_grid = np.linspace(0, 2000, n_points)
        scaling = self._polymer_scaling(gamma)
        if getattr(self, "model_kind", None) == "density_scaled_nu":
            total_sup = phi_r + 1.62 * phi_r**(3 * np.asarray(self.gamma_Sup_p))
            total_drop = phi_r + 1.62 * phi_r**(3 * np.asarray(self.gamma_Drop_p))
            ax.plot(phi_r * x_scale, y_scale * total_sup,
                    label=plabel + " supernatant", color=pcolor)
            ax.plot(phi_r * x_scale, y_scale * total_drop,
                    label=plabel + " droplet", color=pcolor, linestyle="--")
            ax.scatter(phi_r[-1] * x_scale, y_scale * total_sup[-1],
                       color=pcolor, s=150, marker="o")
        else:
            total = scaling.pressure(phi_grid)
            polymer = total - phi_grid
            ax.plot(phi_grid * x_scale, y_scale * total, label=plabel, color=pcolor)
            ax.plot(phi_grid * x_scale, y_scale * polymer, color=pcolor, linestyle="--", linewidth=1)

            if highlight:
                idx = np.abs(np.asarray(self.Pol_Sup_p) - plim).argmin()
                phi_highlight = phi_r[idx:]
                total_highlight = scaling.pressure(phi_highlight)
                ax.plot(phi_highlight * x_scale, y_scale * total_highlight,
                        color=pcolor, linewidth=10, alpha=0.6)
                ax.scatter(
                    phi_r[-1] * x_scale,
                    y_scale * scaling.pressure(phi_r[-1]),
                    color=pcolor,
                    s=150,
                    marker="o",
                )

        if secondary_axis is None:
            secondary_axis = not (mass or reduced)
        if secondary_axis:
            rb = 8.314
            temperature = 273.15 + self.TC

            def inverse(x):
                return 1000 * x / rb / temperature

            def forward(x):
                return x * rb * temperature / 1000

            secay = ax.secondary_yaxis("right", functions=(forward, inverse))
            secay.set_ylabel(r"$\Pi$ (kPa)", fontsize=28)
            secay.tick_params(axis="y", labelsize=24)
        return ax

    def a_plot(
        self,
        pcolor,
        plabel,
        ax=None,
        *,
        show_hard_sphere=True,
        linewidth=3,
    ):
        """Plot free volume fraction against protein concentration."""
        self._require_results()
        ax = self._get_axis(ax)
        q = self.Rpol_in / self.Rpr_in
        phi_bsa_sup = np.asarray(self.Pr_Sup_p) / self.phix_bsa

        ax.scatter(self.Pr_Drop_p[-1], self.a_Drop_p[-1], color=pcolor, s=150, marker="o")
        ax.plot(self.Pr_Drop_p, self.a_Drop_p, label=plabel, color=pcolor, linewidth=linewidth)
        ax.plot(self.Pr_Sup_p, self.a_Sup_p, color=pcolor, linewidth=linewidth)

        if show_hard_sphere:
            phi_bsa_range = np.linspace(0, 0.6, 1000)
            ax.plot(
                self.Pr_Sup_p,
                [1 - phi * (1 + qx) ** 3 for qx, phi in zip(self.qx_Sup_p, phi_bsa_sup)],
                color=pcolor,
                linewidth=2,
                linestyle="--",
            )
            ax.plot(
                phi_bsa_range * self.phix_bsa,
                [1 - phi * (1 + q) ** 3 for phi in phi_bsa_range],
                color=pcolor,
                linewidth=1,
                linestyle="--",
            )
        return ax

    def a_plot_peg(self, pcolor, plabel, ax=None, *, linewidth=3):
        """Plot free volume fraction against polymer concentration."""
        self._require_results()
        ax = self._get_axis(ax)
        ax.scatter(self.Pol_Drop_p[-1], self.a_Drop_p[-1], color=pcolor, s=150, marker="o")
        ax.plot(self.Pol_Drop_p, self.a_Drop_p, label=plabel, color=pcolor, linewidth=linewidth)
        ax.plot(self.Pol_Sup_p, self.a_Sup_p, color=pcolor, linewidth=linewidth)
        return ax

    def g_plot(self, pcolor, plabel, ax=None, *, gamma=None):
        """Plot the polymer contribution weighted by g."""
        self._require_results()
        ax = self._get_axis(ax)
        if getattr(self, "model_kind", None) == "density_scaled_nu":
            phi_r = np.asarray(self.phi_R)
            pressure_drop = self.phix_peg * (
                phi_r + 1.62 * phi_r**(3 * np.asarray(self.gamma_Drop_p)))
            pressure_sup = self.phix_peg * (
                phi_r + 1.62 * phi_r**(3 * np.asarray(self.gamma_Sup_p)))
        else:
            pressure_drop = pressure_sup = (
                self.phix_peg * self._polymer_scaling(gamma).pressure(self.phi_R))
        ax.semilogx(self.Pr_Drop_p, pressure_drop * self.g_Drop,
                    label=plabel, color=pcolor, linestyle="--", linewidth=1)
        ax.semilogx(self.Pr_Sup_p, pressure_sup * self.g_Sup,
                    color=pcolor, linestyle="--", linewidth=1)
        ax.scatter(self.Pr_Drop_p[-1], pressure_drop[-1] * self.g_Drop[-1],
                   color=pcolor, s=150, marker="o")
        return ax

    def h_plot(self, pcolor, plabel, ax=None):
        """Plot h against protein concentration."""
        self._require_results()
        ax = self._get_axis(ax)
        ax.scatter(self.Pr_Drop_p[-1], self.h_Drop[-1], color=pcolor, s=150, marker="o")
        ax.plot(self.Pr_Drop_p, self.h_Drop, label=plabel, color=pcolor)
        ax.plot(self.Pr_Sup_p, self.h_Sup, color=pcolor, linewidth=1, linestyle="--")
        return ax

    def mu_plot(
        self,
        pcolor,
        plabel,
        ax=None,
        plim=500,
        *,
        depletion=False,
        reference_band=False,
    ):
        """Plot chemical potential curves."""
        self._require_results()
        ax = self._get_axis(ax)
        idx = np.abs(np.asarray(self.Pol_Sup_p) - plim).argmin()

        if depletion:
            mu_drop = np.asarray(self.Mu_Drop) - np.asarray(self.Mu0_Drop)
            mu_sup = np.asarray(self.Mu_Sup) - np.asarray(self.Mu0_Sup)
            ax.scatter(self.Pr_Drop_p[-1], mu_drop[-1], color=pcolor, s=150, marker="o")
            ax.semilogx(self.Pr_Drop_p, mu_drop, label=plabel, color=pcolor)
            ax.semilogx(self.Pr_Sup_p, mu_sup, color=pcolor, linewidth=1, linestyle="--")
            if reference_band:
                tmpx = np.linspace(1e-3, 10, 1000)
                tmpy = [12 - 6 * np.log(tmp) for tmp in tmpx]
                ax.semilogx(tmpx, tmpy, color="gray", linewidth=200, alpha=0.05)
            return ax

        ax.plot(self.Pr_Sup_p, self.Mu_Sup, label=plabel + " total supernatant", color=pcolor)
        ax.plot(self.Pr_Drop_p, self.Mu_Drop, label=plabel + " total droplet", color=pcolor)
        ax.plot(self.Pr_Sup_p[idx:], self.Mu_Sup[idx:], color=pcolor, linewidth=10, alpha=0.6)
        ax.plot(self.Pr_Drop_p[idx:], self.Mu_Drop[idx:], color=pcolor, linewidth=10, alpha=0.6)
        ax.scatter(self.Pr_Sup_p[-1], self.Mu_Sup[-1], color=pcolor, marker="*", s=200)
        ax.plot(self.Pr_Sup_p, self.Mu0_Sup, label=plabel + " HS supernatant", color=pcolor, linestyle="--")
        ax.plot(self.Pr_Drop_p, self.Mu0_Drop, label=plabel + " HS droplet", color=pcolor, linestyle="--")
        return ax

    def Pi_Mu_a_plots(self, axes, pcolor, plabel, mpeg=None, *, verbose=False):
        """Plot pressure, chemical potential, and free-volume summary panels."""
        self._require_results()
        q = self.Rpol_in / self.Rpr_in
        if verbose:
            print(self.phix_bsa)
            print(self.phix_peg)
            print(f"rescale with q = {q:.2f}, phix_peg/q**3={self.phix_peg * q**3:.2f}")

        axes[0].plot(self.Pr_Drop_p, self.Pi_Drop / 1000, label="Droplet Total", color=pcolor)
        axes[0].plot(self.Pr_Drop_p, (self.Pi_Drop - self.P0_Drop) / 1000, label="Polymer", linestyle="--", color=pcolor)
        axes[0].plot(self.Pr_Drop_p, self.P0_Drop / 1000, label="Droplet HS", linestyle=":", color=pcolor)
        axes[0].plot(self.Pr_Sup_p, self.Pi_Sup / 1000, label="Supernatant Total", color=pcolor)
        axes[0].plot(self.Pr_Sup_p, (self.Pi_Sup - self.P0_Sup) / 1000, label="Polymer", linestyle="--", color=pcolor)
        axes[0].plot(self.Pr_Sup_p, self.P0_Sup / 1000, label="Sup HS", linestyle=":", color=pcolor)
        axes[0].set_title(r"$\Pi$ (kPa)")
        axes[0].set_xlabel(r"$[BSA]_{sup}$ (mM)")
        axes[0].set_ylabel(r"$\Pi$ (kPa)")
        axes[0].set_xlim(0, 7)
        axes[0].set_ylim(0, 300)

        axes[1].plot(self.Pr_Sup_p, self.Mu_Sup, label="Mu Total Supernatant", color=pcolor)
        axes[1].plot(self.Pr_Drop_p, self.Mu_Drop, label="Mu Total Drop", color=pcolor)
        axes[1].plot(self.Pr_Sup_p, self.MuR_Sup, label="Mu Depletion Supernatant", color=pcolor, linestyle="--")
        axes[1].plot(self.Pr_Drop_p, self.MuR_Drop, label="Mu Depletion Drop", color=pcolor, linestyle="--")
        axes[1].plot(self.Pr_Sup_p, self.Mu0_Sup, label="Mu HS Supernatant", color=pcolor, linestyle="-.")
        axes[1].plot(self.Pr_Drop_p, self.Mu0_Drop, label="Mu HS Drop", color=pcolor, linestyle="-.")
        axes[1].set_title("Chemical Potential")
        axes[1].set_xlabel(r"$[BSA]_{sup}$ (mM)")
        axes[1].set_ylabel(r"$\mu_{BSA}/kT$")
        axes[1].set_xlim(0, 8)
        axes[1].set_ylim(-100, 50)
        axes[1].set_xscale("log")

        # Plot the actual SPT free volume. The former c_PEG*q**1.63*qx**1.5
        # ordinate was an ad hoc concentration rescaling, not alpha.
        axes[2].plot(self.Pr_Drop_p, self.a_Drop_p, label="Droplet alpha", linestyle="-", color=pcolor)
        axes[2].plot(self.Pr_Sup_p, self.a_Sup_p, label="Supernatant alpha", linestyle="-", color=pcolor)
        axes[2].set_title("Free volume fraction")
        axes[2].set_xlabel(r"$[BSA]$ (mM)")
        axes[2].set_ylabel("Free volume fraction")
        return axes

    def to_legacy_list(self) -> list[Any]:
        """Return results in the notebook's original 32-item list order."""
        if getattr(self, "_rod_legacy_ambiguous", False):
            raise ValueError(
                "This rod phase diagram has multiple or disconnected branches; "
                "use rod_branches or calculate one continuous coexistence branch "
                "before converting to the legacy list."
            )
        return [getattr(self, field) for field in RETURN_FIELDS]

    def _build_params(self) -> list[Any]:
        return [
            self.Rpr_in,
            self.Rpol_in,
            self.mbsa,
            self.mpeg,
            self.solvent,
            self.N_steps,
            self.PhiP_min,
            self.PhiP_max,
            self.outguess_c,
            self.outguess_tp,
            self.guess_b,
            self.TC,
            self.phix_bsa,
            self.phix_peg,
        ]

    def _reject_kuhn_variant(self):
        if getattr(self, "kuhn_crossover", None) is not None:
            raise ValueError("Kuhn crossover supports spherical GFVT/GFVT_full/GFVT_es only; "
                             "it cannot be combined with this variant or empirical overlay.")

    def _clear_scaled_rod_metadata(self) -> None:
        """Discard shape/exclusion metadata when replacing a calculated model."""
        for name in (
            "reference_cp_phi", "reference_outguess_c", "critical_kind", "onset",
            "Phi_Drop_p", "Phi_Sup_p", "L_Drop_p", "L_Sup_p",
            "vbsa_Drop_p", "vbsa_Sup_p", "phix_bsa_Drop_p", "phix_bsa_Sup_p",
            "boundary_Drop_p", "corner_fraction_p", "model_kind",
            "reference_cp", "reference_tp", "reference_outguess_tp",
            "reference_cp_list", "reference_tp_list",
            "es_m", "es_Z", "es_ionic_strength", "es_bjerrum_length",
            "es_kappa", "es_d_eff", "es_depletion_exponent",
            "exclusion_width", "exclusion_law", "coexistence_breaks_p", "critical_endpoint",
            "nu_scale_law", "nu_scale_width", "nu_Drop_p", "nu_Sup_p",
            "gamma_Drop_p", "gamma_Sup_p",
            "rod_theory", "dep_model", "odf", "L_rod", "sigma",
            "rod_branches", "rod_critical_points", "rod_triple_points",
            "rod_requested_coexistence", "_rod_legacy_ambiguous",
            "rod_orientation_grid", "L_over_D", "rod_branch_status",
        ):
            self.__dict__.pop(name, None)

    def _assign_return_values(self, values: list[Any]) -> None:
        for field, value in zip(RETURN_FIELDS, values):
            setattr(self, field, value)
        self.params = values[-1]
        self._sync_aliases()

    def _sync_aliases(self) -> None:
        self.BSA_Drop = self.Pr_Drop_p
        self.PEG_Drop = self.Pol_Drop_p
        self.BSA_Sup = self.Pr_Sup_p
        self.PEG_Sup = self.Pol_Sup_p
        self.a_Drop = self.a_Drop_p
        self.a_Sup = self.a_Sup_p

    @staticmethod
    def _get_axis(ax):
        if ax is not None:
            return ax
        import matplotlib.pyplot as plt

        return plt.gca()

    @staticmethod
    def _maybe_legend(ax, show_legend: bool) -> None:
        if show_legend:
            ax.legend(loc=(1.1, 0), fontsize="x-large")

    @staticmethod
    def _nonzero_data_points(x, y, xerr, yerr):
        x = np.asarray(x)
        y = np.asarray(y)
        xerr = np.broadcast_to(np.asarray(xerr), x.shape)
        yerr = np.broadcast_to(np.asarray(yerr), y.shape)
        mask = (x != 0) & (y != 0)
        return x[mask], y[mask], xerr[mask], yerr[mask], mask

    @staticmethod
    def _parse_binodal_data(
        data,
        *,
        mbsa=0,
        mpeg=0,
        phix_bsa=None,
        phix_peg=None,
        peg_scale=1,
    ):
        data = np.asarray(data)
        if data.shape[1] < 16:
            raise ValueError("Expected binodal data with at least 16 columns.")

        bsa_scale = 1
        peg_mass_scale = peg_scale
        if mbsa != 0 and mpeg != 0:
            bsa_scale = 1 / mbsa
            peg_mass_scale = peg_scale / mpeg
            if phix_bsa is not None and phix_peg is not None:
                bsa_scale /= phix_bsa
                peg_mass_scale /= phix_peg

        return (
            data[:, 0] * bsa_scale,
            data[:, 1] * bsa_scale,
            data[:, 2] * bsa_scale,
            data[:, 3] * bsa_scale,
            data[:, 4] * bsa_scale,
            data[:, 5] * bsa_scale,
            data[:, 6] * peg_mass_scale,
            data[:, 7] * peg_mass_scale,
            data[:, 8] * peg_mass_scale,
            data[:, 9] * peg_mass_scale,
            data[:, 10] * peg_mass_scale,
            data[:, 11] * peg_mass_scale,
            data[:, 12] * bsa_scale,
            data[:, 13] * bsa_scale,
            data[:, 14] * peg_mass_scale,
            data[:, 15] * peg_mass_scale,
        )

    @staticmethod
    def _plot_data_errorbars(
        ax,
        plabel,
        pcolor,
        BN_BSA_tot,
        BN_BSA_tot_err,
        BN_BSA_sup,
        BN_BSA_sup_err,
        BN_BSA_drop,
        BN_BSA_drop_err,
        BN_PEG_tot,
        BN_PEG_tot_err,
        BN_PEG_sup,
        BN_PEG_sup_err,
        BN_PEG_drop,
        BN_PEG_drop_err,
        CP_BSA,
        CP_BSA_err,
        CP_PEG,
        CP_PEG_err,
        *,
        n_tie=False,
        plot_tot=False,
    ) -> None:
        BN_BSA_sup_all = np.asarray(BN_BSA_sup)
        BN_PEG_sup_all = np.asarray(BN_PEG_sup)
        BN_BSA_drop_all = np.asarray(BN_BSA_drop)
        BN_PEG_drop_all = np.asarray(BN_PEG_drop)

        (
            BN_BSA_sup,
            BN_PEG_sup,
            BN_BSA_sup_err,
            BN_PEG_sup_err,
            sup_mask,
        ) = PD._nonzero_data_points(BN_BSA_sup, BN_PEG_sup, BN_BSA_sup_err, BN_PEG_sup_err)
        (
            BN_BSA_drop,
            BN_PEG_drop,
            BN_BSA_drop_err,
            BN_PEG_drop_err,
            drop_mask,
        ) = PD._nonzero_data_points(BN_BSA_drop, BN_PEG_drop, BN_BSA_drop_err, BN_PEG_drop_err)
        (
            CP_BSA,
            CP_PEG,
            CP_BSA_err,
            CP_PEG_err,
            _,
        ) = PD._nonzero_data_points(CP_BSA, CP_PEG, CP_BSA_err, CP_PEG_err)
        (
            BN_BSA_tot,
            BN_PEG_tot,
            BN_BSA_tot_err,
            BN_PEG_tot_err,
            _,
        ) = PD._nonzero_data_points(BN_BSA_tot, BN_PEG_tot, BN_BSA_tot_err, BN_PEG_tot_err)

        ax.errorbar(
            BN_BSA_sup,
            BN_PEG_sup,
            xerr=BN_BSA_sup_err,
            yerr=BN_PEG_sup_err,
            markersize=12,
            fmt="o",
            c=pcolor,
            ecolor=pcolor,
            capsize=5,
            capthick=1,
            label=plabel + " binodal",
        )
        ax.errorbar(
            BN_BSA_drop,
            BN_PEG_drop,
            xerr=BN_BSA_drop_err,
            yerr=BN_PEG_drop_err,
            markersize=12,
            fmt="o",
            c=pcolor,
            ecolor=pcolor,
            capsize=5,
            capthick=1,
        )
        ax.errorbar(
            CP_BSA,
            CP_PEG,
            xerr=CP_BSA_err,
            yerr=CP_PEG_err,
            markersize=12,
            fmt="X",
            c=pcolor,
            ecolor=pcolor,
            capsize=5,
            capthick=1,
            label=plabel + " critical point",
        )
        if plot_tot:
            ax.errorbar(
                BN_BSA_tot,
                BN_PEG_tot,
                xerr=BN_BSA_tot_err,
                yerr=BN_PEG_tot_err,
                markersize=12,
                fmt="o",
                c=pcolor,
                ecolor=pcolor,
                capsize=5,
                capthick=1,
                label=plabel + " binodal",
            )
        if n_tie:
            tie_mask = sup_mask & drop_mask
            tie_bsa_sup = BN_BSA_sup_all[tie_mask]
            tie_peg_sup = BN_PEG_sup_all[tie_mask]
            tie_bsa_drop = BN_BSA_drop_all[tie_mask]
            tie_peg_drop = BN_PEG_drop_all[tie_mask]
            for idx in range(len(tie_bsa_sup)):
                ax.plot(
                    [tie_bsa_sup[idx], tie_bsa_drop[idx]],
                    [tie_peg_sup[idx], tie_peg_drop[idx]],
                    "--",
                    color=pcolor,
                )

    def _require_results(self) -> None:
        if self.phi_R is None:
            raise RuntimeError("Run GFVT() first, or create the object with PD.from_legacy_list(...).")

    @staticmethod
    def _reject_fvt_overlay(FVT_color) -> None:
        if FVT_color:
            raise NotImplementedError("FVT overlays require FVT output and are not supported by the PD class.")

    def _plot_masses(self, mbsa, mpeg):
        return (self.mbsa if mbsa == 0 else mbsa, self.mpeg if mpeg == 0 else mpeg)

    def _get_phix_scales(self) -> tuple[float, float]:
        if self.phix_bsa is not None and self.phix_peg is not None:
            return float(self.phix_bsa), float(self.phix_peg)

        vbsa = 4 / 3 * np.pi * self.Rpr_in**3 * 6.022e23
        vpeg = 4 / 3 * np.pi * self.Rpol_in**3 * 6.022e23
        return 1 / vbsa, 1 / vpeg

    def _full_label(self, plabel) -> str:
        if getattr(self, "geometry", "spheres") == "spherocylinders":
            return (plabel + f"rods, L/D = {self.L_rod / self.D:g}, "
                    f"sigma/D = {self.sigma / self.D:g}, {self.dep_model}, {self.odf}")
        cross = getattr(self, "kuhn_crossover", None)
        if cross is not None:
            plabel += f"Kuhn crossover s={cross.s:g}, b={cross.b*1e9:g} nm, m={cross.m:g}, "
        if getattr(self, "model_kind", None) == "electrostatic":
            return plabel + f"GFVT_es, m = {self.es_m:.3g}, q = {self.q:.2f}, R = {self.Rpr_in * 1e9:.2f} nm"
        if getattr(self, "geometry", "spheres") == "isotropic_rods_scaled":
            return plabel + f"isotropic rods, D = {self.D * 1e9:.2f} nm, scaled L (s = {self.s:g}), Rp = {self.Rpol_in * 1e9:.2f} nm"
        if getattr(self, "geometry", "spheres") == "isotropic_rods":
            return plabel + f"isotropic rods, D = {self.D * 1e9:.2f} nm, L = {self.L * 1e9:.2f} nm, Rp = {self.Rpol_in * 1e9:.2f} nm"
        return plabel + f"q = {self.q:.1f}, Rbsa = {self.Rpr_in * 1e9:.2f} nm, Rp = {self.Rpol_in * 1e9:.2f} nm"

    def _protein_volume_fractions(self):
        """Return occupied fractions using each phase's actual particle volume."""
        if getattr(self, "geometry", "spheres") == "isotropic_rods_scaled":
            return np.asarray(self.Phi_Drop_p), np.asarray(self.Phi_Sup_p)
        phix_bsa, _ = self._get_phix_scales()
        return np.asarray(self.Pr_Drop_p) / phix_bsa, np.asarray(self.Pr_Sup_p) / phix_bsa

    def _drop_line_coordinates(self, x, y, *, start=0):
        """Do not connect discontinuous corner/interior droplet branches."""
        breaks = getattr(self, "coexistence_breaks_p", None)
        if breaks is not None:
            cuts = np.flatnonzero(np.asarray(breaks, dtype=bool)[start:]) + 1
            cuts = cuts[cuts < len(x)]
            return np.insert(np.asarray(x, dtype=float), cuts, np.nan), np.insert(np.asarray(y, dtype=float), cuts, np.nan)
        boundary = getattr(self, "boundary_Drop_p", None)
        has_corner = (getattr(self, "geometry", "spheres") == "isotropic_rods_scaled"
                      or getattr(self, "model_kind", None) == "scaled_exclusion_layer")
        if has_corner and boundary is not None:
            cuts = np.flatnonzero(np.diff(np.asarray(boundary, dtype=int)[start:])) + 1
            return np.insert(np.asarray(x, dtype=float), cuts, np.nan), np.insert(np.asarray(y, dtype=float), cuts, np.nan)
        return x, y

    def _plot_binodal_lines(self, ax, x_drop, y_drop, x_sup, y_sup, pcolor, full_label) -> None:
        drop_x, drop_y = self._drop_line_coordinates(x_drop, y_drop)
        ax.plot(drop_x, drop_y, "-", color=pcolor, ms=2, label=full_label + " binodal")
        ax.plot(x_sup, y_sup, "-", color=pcolor, ms=2)

    @staticmethod
    def _plot_tie_lines(ax, x_drop, y_drop, x_sup, y_sup, pcolor, n_tie) -> None:
        if n_tie <= 0:
            return
        for idx in range(0, len(x_drop), n_tie):
            ax.plot([x_drop[idx], x_sup[idx]], [y_drop[idx], y_sup[idx]], "--", color=pcolor)

    def _plot_critical_lines(self, ax, crit_color, full_label, *, mass=False, mbsa=1, mpeg=1) -> None:
        if not crit_color:
            return

        phix_bsa, phix_peg = self._get_phix_scales()

        if self.cp_list is not None and len(self.cp_list) > 0:
            cp_list = np.asarray(self.cp_list)
            q_clist, Rpr_clist, Rpol_clist, phix_peg_clist, crit1_clist, crit2_clist, Pr_Crit_clist, Pol_Crit_clist = cp_list.T
            x = Pr_Crit_clist * mbsa if mass else Pr_Crit_clist
            y = Pol_Crit_clist * mpeg if mass else Pol_Crit_clist
            ax.plot(x, y, "--", color=crit_color, ms=2, label=full_label + " critical line")

            if self.tp_list is None or len(self.tp_list) == 0:
                return
            tp_list = np.asarray(self.tp_list)
            _, _, _, _, tp1, tp2, tp3, tp4, Pr_tp1, Pr_tp2, Pr_tp3, Pol_tp1, Pol_tp2, Pol_tp3 = tp_list.T
            for idx, (tp_x, tp_y) in enumerate(
                (
                    (Pr_tp1, Pol_tp1),
                    (Pr_tp2, Pol_tp2),
                    (Pr_tp3, Pol_tp3),
                ),
                start=1,
            ):
                x = tp_x * mbsa if mass else tp_x
                y = tp_y * mpeg if mass else tp_y
                ax.plot(x, y, "-", color=crit_color, markersize=6, label=full_label + f" triple line {idx}")
        
        if self.tp is None or len(self.tp) < 6:
            return

        ax.scatter(
            [t * (mbsa if mass else 1) for t in self.tp[0:3]],
            [t * (mpeg if mass else 1) for t in self.tp[3:6]],
            color=crit_color,
            marker="o",
            s=60,
            label=full_label + f" triple point",
            )    
            

    def _plot_phi_critical_lines(self, ax, crit_color, full_label, *, gamma) -> None:
        if not crit_color:
            return

        if gamma is None:
            qexp = 0
        else:
            qexp = -1 / gamma

        phix_bsa, phix_peg = self._get_phix_scales()

        if self.cp_list is not None and len(self.cp_list) > 0:
            cp_list = np.asarray(self.cp_list)
            q_clist, Rpr_clist, Rpol_clist, phix_peg_clist, crit1_clist, crit2_clist, Pr_Crit_clist, Pol_Crit_clist = cp_list.T
            ax.plot(
                [pr / phix_bsa for pr in Pr_Crit_clist],
                [q_item ** qexp * pol / phix for pol, phix, q_item in zip(Pol_Crit_clist, phix_peg_clist, q_clist)],
                "--",
                color=crit_color,
                ms=2,
                label=full_label + " critical line",
            )

            if self.tp_list is None or len(self.tp_list) == 0:
                return
            tp_list = np.asarray(self.tp_list)
            q_tlist, _, _, phix_peg_tlist, tp1, tp2, tp3, tp4, Pr_tp1, Pr_tp2, Pr_tp3, Pol_tp1, Pol_tp2, Pol_tp3 = tp_list.T
            for idx, (tp_x, tp_y) in enumerate(
                (
                    (Pr_tp1, Pol_tp1),
                    (Pr_tp2, Pol_tp2),
                    (Pr_tp3, Pol_tp3),
                ),
                start=1,
            ):
                ax.plot(
                    tp_x / phix_bsa,
                    [q_item ** qexp * pol / phix for pol, phix, q_item in zip(tp_y, phix_peg_tlist, q_tlist)],
                    color=crit_color,
                    label=full_label + f" triple line {idx}",
                )

        if self.tp is None or len(self.tp) < 6:
            return

        ax.scatter(
            [t / phix_bsa for t in self.tp[0:3]],
            [self.q ** qexp * t / phix_peg for t in self.tp[3:6]],
            color=crit_color,
            marker="o",
            s=60,
            label=full_label + f" triple point",
            )

    def _polymer_scaling(self, gamma=None):
        scaling = PolymerScaling(self.solvent, getattr(self, "nu", None),
                                 getattr(self, "kuhn_crossover", None), self.Rpr_in)
        if gamma is not None:
            gamma = self._axis_gamma(gamma)
            if gamma is None or not np.isclose(gamma, scaling.gamma, rtol=1e-10, atol=0):
                raise ValueError("gamma must match the calculated model; set nu on PD and recalculate to change polymer scaling.")
        return scaling

    def _axis_gamma(self, gamma):
        # Numeric gamma rescales coordinates only. True uses the model value.
        if gamma is None or gamma is False:
            return None
        if gamma is True:
            return self._polymer_scaling().gamma
        if not np.isscalar(gamma) or not np.isreal(gamma) or not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be positive and finite, True, or None.")
        return float(gamma)

    def _qxf(self, q, Phi_pol):
        if getattr(self, "model_kind", None) == "electrostatic":
            from .electrostatics import ElectrostaticEOS
            return ElectrostaticEOS(
                q, self.es_m, depletion_exponent=self.es_depletion_exponent,
                nu=getattr(self, "nu", None),
                kuhn_crossover=getattr(self, "kuhn_crossover", None), protein_radius=self.Rpr_in,
            ).depletion(Phi_pol)
        return self._polymer_scaling().qxf(q, Phi_pol)

    def _Qsf(self, Phi, Phi_pol):
        y = Phi / (1 - Phi)
        qx, dPidPhiP = self._qxf(self.q, Phi_pol)
        a = 3 * qx + 3 * qx**2 + qx**3
        b = 9 / 2 * qx**2 + 3 * qx**3
        c = 3 * qx**3
        return a, b, c, y, dPidPhiP

    def _alpha(self, Phi, Phi_pol):
        if getattr(self, "model_kind", None) == "electrostatic":
            from .electrostatics import ElectrostaticEOS
            eos = ElectrostaticEOS(self.q, self.es_m, depletion_exponent=self.es_depletion_exponent,
                                   nu=getattr(self, "nu", None),
                                   kuhn_crossover=getattr(self, "kuhn_crossover", None),
                                   protein_radius=self.Rpr_in)
            return eos.alpha_terms(Phi, eos.depletion(Phi_pol)[0], order=0)
        if getattr(self, "model_kind", None) == "scaled_exclusion_layer" and self.s > 0:
            offset = np.asarray(Phi) - self.reference_cp_phi
            width = getattr(self, "exclusion_width", 0.0)
            if width > 0:
                from .smooth_exclusion import smooth_ramp
                increment = smooth_ramp(offset, width, order=0)[0]
            else:
                increment = np.maximum(offset, 0)
            qx = self._qxf(self.q, Phi_pol)[0] + self.s * increment
            return alpha_spt(Phi, qx)
        if getattr(self, "model_kind", None) == "density_scaled_nu" and self.s > 0:
            from .nu_scale import density_scaled_qxf
            qx = density_scaled_qxf(
                self.q, Phi_pol, Phi, nu=self.nu, s=self.s,
                phi_c=self.reference_cp_phi, width=self.nu_scale_width,
            )[0]
            return alpha_spt(Phi, qx)
        a, b, c, y, _ = self._Qsf(Phi, Phi_pol)
        return (1 - Phi) * np.exp(-(a * y + b * y**2 + c * y**3))

    def _analytical_base_curves(self):
        """Fleer--Tuinier Eqs. 6.47/6.48: empirical display approximations.

        The 0.4/0.25 shape fits are not polymer exponents or critical
        asymptotes, and never enter the numerical coexistence solver.
        Their accuracy is not established for arbitrary user-supplied nu.
        """
        self._reject_kuhn_variant()
        if self.outguess_tp is None:
            raise ValueError("Analytical sphere/triple-point overlays are unavailable for this model; use molplot or molplot_phi.")
        c_bsa, c_PEG = self.outguess_c
        tp_bsa1, tp_bsa2, tp_bsa3, tp_PEG = self.outguess_tp
        y_a_L = np.linspace(c_PEG, tp_PEG, 2000)
        q = self.q

        scaling = self._polymer_scaling()
        gamma, c2 = scaling.gamma, scaling.zeta
        qq = q ** (-1 / gamma)
        Pv_a_L = scaling.pressure(y_a_L) / q**3
        q_a_G = scaling.depletion(q, y_a_L)

        bsa_a_L = c_bsa + (tp_bsa2 - c_bsa) * ((Pv_a_L - Pv_a_L[0]) / (Pv_a_L[-1] - Pv_a_L[0])) ** 0.4
        bsa_a_G = c_bsa + (tp_bsa1 - c_bsa) * ((q_a_G - q_a_G[0]) / (q_a_G[-1] - q_a_G[0])) ** 0.25
        return gamma, c2, qq, y_a_L, bsa_a_L, bsa_a_G
